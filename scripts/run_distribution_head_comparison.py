#!/usr/bin/env python3
"""
Per-city distribution-head comparison (Phase 2 deliverable #6).

Trains three forecast heads on a city's processed (z-scored) features —
Gaussian, 7-quantile, and 2-component Gaussian mixture — selects the best by
validation CRPS, and reports the held-out OOS CRPS and (where real Kalshi
contracts overlap the test window) contract Brier for each.

Output: results/<city>/diagnostics/distribution_heads.json

This is a model-selection diagnostic, not a promotion gate: it answers "which
output parameterization fits this city's residual structure best?" before the
hparam/ensemble work commits to one.

Usage:
    python scripts/run_distribution_head_comparison.py --city chi
    python scripts/run_distribution_head_comparison.py --all --epochs 150
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.bucket_semantics import bucket_outcome_from_edges  # noqa: E402
from src.city_config import get_city_config  # noqa: E402
from src.bucket_semantics import (  # noqa: E402
    bucket_prob_from_edges,
    mixture_bucket_prob_from_edges,
)
from src.distribution_heads import (  # noqa: E402
    DEFAULT_QUANTILE_LEVELS,
    HeadScore,
    build_gaussian_net,
    build_mixture_net,
    build_quantile_net,
    gaussian_crps,
    gaussian_mixture_crps,
    gaussian_nll_torch,
    mixture_nll_torch,
    pinball_loss_torch,
    quantile_bucket_prob_from_edges,
    quantile_crps,
    select_best_head,
)
from src.model_diagnostics import load_presettlement  # noqa: E402
from scripts.run_model_diagnostics import PRESETTLEMENT  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

ALL_CITIES = ["chi", "phl", "atl", "aus"]
CITY_TARGET = {c: f"{c.upper()}_TMAX" for c in ALL_CITIES}


def _load_split(processed: Path, split: str, target_col: str):
    X = pd.read_csv(processed / f"features_{split}.csv", index_col=0, parse_dates=True)
    y = pd.read_csv(processed / f"target_{split}.csv", index_col=0, parse_dates=True)
    X = X.dropna(axis=1, how="all").fillna(0.0)
    return X, y.iloc[:, 0].rename(target_col)


def _align_columns(X_train, X_val, X_test):
    cols = X_train.columns
    return X_train, X_val.reindex(columns=cols, fill_value=0.0), X_test.reindex(
        columns=cols, fill_value=0.0
    )


def _train(net, loss_fn, X_tr, y_tr, X_va, y_va, epochs, lr=1e-2,
           patience=40, batch_size=256):
    """Mini-batch Adam with val-loss early stopping and gradient clipping."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    n = X_tr.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(X_tr[idx]), y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(X_va), y_va).item()
        sched.step(val)
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def _contract_groups(city_code: str, test_dates: pd.DatetimeIndex, y_test: np.ndarray):
    """Group test-window contract rows by (lo, hi) for vectorized Brier scoring.

    Returns a list of ``(lo, hi, positions, outcomes)`` where ``positions``
    indexes into the test-day arrays and ``outcomes`` is the settled YES/NO for
    each contract row, or ``None`` when no presettlement file overlaps.
    """
    pre_path = PROJECT_ROOT / "data" / PRESETTLEMENT.get(city_code, "")
    if not pre_path.name or not pre_path.exists():
        return None
    contracts = load_presettlement(pre_path)
    date_to_pos = {d.normalize(): i for i, d in enumerate(test_dates)}
    rows = contracts[contracts["date"].isin(set(date_to_pos))]
    if rows.empty:
        return None
    groups = []
    for (lo, hi), grp in rows.groupby(["threshold_low", "threshold_high"]):
        pos = grp["date"].map(date_to_pos).to_numpy()
        outcomes = bucket_outcome_from_edges(y_test[pos], float(lo), float(hi))
        groups.append((float(lo), float(hi), pos, outcomes.astype(float)))
    return groups


def _contract_brier(prob_fn, groups) -> float:
    """Mean contract Brier over all rows; ``prob_fn(lo, hi, pos) -> probs``."""
    if not groups:
        return float("nan")
    probs, outs = [], []
    for lo, hi, pos, outcome in groups:
        probs.append(prob_fn(lo, hi, pos))
        outs.append(outcome)
    p, o = np.concatenate(probs), np.concatenate(outs)
    return float(np.mean((p - o) ** 2))


def run_city(city_code: str, epochs: int = 150, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = get_city_config(city_code)
    processed = Path(cfg.data_dir) / "processed"
    target_col = CITY_TARGET[city_code]

    X_tr, y_tr = _load_split(processed, "train", target_col)
    X_va, y_va = _load_split(processed, "val", target_col)
    X_te, y_te = _load_split(processed, "test", target_col)
    X_tr, X_va, X_te = _align_columns(X_tr, X_va, X_te)
    n_features = X_tr.shape[1]

    # Center the target so the Gaussian mu head starts near the mean (Phase 0 fix).
    y_mean = float(y_tr.mean())
    to_t = lambda df: torch.tensor(df.to_numpy(dtype=np.float32))
    Xt, Xv, Xte = to_t(X_tr), to_t(X_va), to_t(X_te)
    yt = torch.tensor((y_tr.to_numpy(dtype=np.float32) - y_mean))
    yv = torch.tensor((y_va.to_numpy(dtype=np.float32) - y_mean))
    y_test = y_te.to_numpy(dtype=float)
    test_dates = pd.DatetimeIndex(X_te.index)

    levels = np.array(DEFAULT_QUANTILE_LEVELS)

    # --- Gaussian head ---
    g_net = build_gaussian_net(n_features)
    _train(g_net, lambda out, y: gaussian_nll_torch(out[0], out[1], y), Xt, yt, Xv, yv, epochs)
    with torch.no_grad():
        mu_g, sig_g = g_net(Xte)
    mu_g = mu_g.numpy() + y_mean
    sig_g = np.maximum(sig_g.numpy(), 1e-3)

    # --- Quantile head ---
    q_net = build_quantile_net(n_features, levels=DEFAULT_QUANTILE_LEVELS)
    _train(q_net, lambda out, y: pinball_loss_torch(DEFAULT_QUANTILE_LEVELS, out, y),
           Xt, yt, Xv, yv, epochs)
    with torch.no_grad():
        qv = q_net(Xte).numpy() + y_mean

    # --- Mixture head ---
    m_net = build_mixture_net(n_features, n_components=2)
    _train(m_net, lambda out, y: mixture_nll_torch(out[0], out[1], out[2], y),
           Xt, yt, Xv, yv, epochs)
    with torch.no_grad():
        w_m, mu_m, sig_m = m_net(Xte)
    w_m = w_m.numpy()
    mus_m = mu_m.numpy() + y_mean
    sig_m = np.maximum(sig_m.numpy(), 1e-3)

    # OOS CRPS per head (the primary probabilistic score).
    g_crps = float(np.mean(gaussian_crps(mu_g, sig_g, y_test)))
    q_crps = float(np.mean(quantile_crps(levels, qv, y_test)))
    m_crps = float(np.mean(gaussian_mixture_crps(w_m, mus_m, sig_m, y_test)))

    # Contract Brier on real Kalshi rows overlapping the test window.
    groups = _contract_groups(city_code, test_dates, y_test)
    g_brier = _contract_brier(
        lambda lo, hi, pos: bucket_prob_from_edges(mu_g[pos], sig_g[pos], lo, hi), groups
    )
    q_brier = _contract_brier(
        lambda lo, hi, pos: quantile_bucket_prob_from_edges(levels, qv[pos], lo, hi), groups
    )
    m_brier = _contract_brier(
        lambda lo, hi, pos: mixture_bucket_prob_from_edges(
            w_m[pos], mus_m[pos], sig_m[pos], lo, hi
        ),
        groups,
    )

    results = {
        "gaussian": {"oos_crps": g_crps, "contract_brier": g_brier},
        "quantile": {"oos_crps": q_crps, "contract_brier": q_brier},
        "mixture": {"oos_crps": m_crps, "contract_brier": m_brier},
    }
    best = select_best_head([
        HeadScore("gaussian", g_crps, g_brier, len(y_test)),
        HeadScore("quantile", q_crps, q_brier, len(y_test)),
        HeadScore("mixture", m_crps, m_brier, len(y_test)),
    ])

    summary = {
        "city_code": city_code,
        "n_features": int(n_features),
        "n_test_days": int(len(y_test)),
        "test_date_min": str(test_dates.min().date()),
        "test_date_max": str(test_dates.max().date()),
        "heads": results,
        "best_head_by_oos_crps": best,
    }

    out_dir = Path(cfg.results_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "distribution_heads.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "[%s] CRPS gaussian=%.3f quantile=%.3f mixture=%.3f -> best=%s",
        city_code, results["gaussian"]["oos_crps"], results["quantile"]["oos_crps"],
        results["mixture"]["oos_crps"], best,
    )
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", choices=ALL_CITIES)
    p.add_argument("--all", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.city and not args.all:
        p.error("provide --city or --all")
    targets = ALL_CITIES if args.all else [args.city]
    failures = []
    for c in targets:
        try:
            run_city(c, epochs=args.epochs, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] head comparison failed: %s", c, exc)
            failures.append(c)
    return 1 if failures and len(failures) == len(targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
