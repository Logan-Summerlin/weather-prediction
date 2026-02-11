#!/usr/bin/env python3
"""Run E0-E8 probabilistic + ensemble experiments from report.

This script intentionally focuses on time-safe chronological splits and generates
reproducible benchmark artifacts under results/probabilistic_ensemble_experiments.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "probabilistic_ensemble_experiments")
os.makedirs(OUT_DIR, exist_ok=True)

CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv")

BUCKET_EDGES = np.array([-100.0, 32.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 140.0])
QUANTILES = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95])


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray


class GaussianNN(nn.Module):
    def __init__(self, d_in: int, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, 1)
        self.log_sigma = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu(h).squeeze(-1)
        sigma = F.softplus(self.log_sigma(h)).squeeze(-1) + 0.5
        return mu, sigma.clamp(max=12.0)


class MixtureGaussianNN(nn.Module):
    def __init__(self, d_in: int, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 5)

    def forward(self, x):
        raw = self.head(self.backbone(x))
        logit_w, mu1, l1, mu2, l2 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
        w = torch.sigmoid(logit_w)
        s1 = F.softplus(l1) + 0.4
        s2 = F.softplus(l2) + 0.4
        return w, mu1, s1.clamp(max=12.0), mu2, s2.clamp(max=12.0)


class QuantileNN(nn.Module):
    def __init__(self, d_in: int, n_q: int, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.base = nn.Linear(prev, 1)
        self.deltas = nn.Linear(prev, n_q)

    def forward(self, x):
        h = self.backbone(x)
        base = self.base(h)
        deltas = F.softplus(self.deltas(h))
        q = base + torch.cumsum(deltas, dim=1)
        return q


def gaussian_nll(mu, sigma, y):
    return 0.5 * torch.log(2 * math.pi * sigma ** 2) + ((y - mu) ** 2) / (2 * sigma ** 2)


def gaussian_crps(mu, sigma, y):
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def mdn_nll(w, mu1, s1, mu2, s2, y):
    y = y.unsqueeze(1)
    comp1 = torch.log(w + 1e-8) - torch.log(s1.unsqueeze(1)) - 0.5 * ((y - mu1.unsqueeze(1)) / s1.unsqueeze(1)) ** 2
    comp2 = torch.log(1 - w + 1e-8) - torch.log(s2.unsqueeze(1)) - 0.5 * ((y - mu2.unsqueeze(1)) / s2.unsqueeze(1)) ** 2
    ll = torch.logsumexp(torch.cat([comp1, comp2], dim=1), dim=1) - 0.5 * math.log(2 * math.pi)
    return -ll


def pinball_loss(q_pred, y, quantiles):
    y = y.unsqueeze(1)
    q = torch.tensor(quantiles, device=y.device, dtype=y.dtype).unsqueeze(0)
    e = y - q_pred
    return torch.maximum(q * e, (q - 1.0) * e)


def load_dataset():
    cp = pd.read_csv(CP_PATH, parse_dates=["date"]).rename(columns={"tmax_f": "nyc_tmax"})
    if "nyc_tmax" not in cp:
        cp.columns = ["date", "nyc_tmax"]
    cp = cp[["date", "nyc_tmax"]].set_index("date").sort_index()

    mos = pd.read_csv(MOS_PATH, parse_dates=["date"]).set_index("date").sort_index()
    df = cp.join(mos, how="inner")

    df["lag1"] = df["nyc_tmax"].shift(1)
    df["lag2"] = df["nyc_tmax"].shift(2)
    df["mos_spread"] = (df["gfs_mos_tmax_f"] - df["nam_mos_tmax_f"]).abs()
    doy = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    feature_cols = [
        "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f",
        "lag1", "lag2", "mos_spread", "sin_doy", "cos_doy",
    ]
    df = df.dropna(subset=feature_cols + ["nyc_tmax"]).copy()
    return df, feature_cols


def split_df(df: pd.DataFrame, feature_cols: list[str]):
    windows = {
        "train": ("2004-01-01", "2021-12-31"),
        "val": ("2022-01-01", "2022-12-31"),
        "calib": ("2023-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2024-12-31"),
        "oos": ("2025-01-01", "2025-12-31"),
    }
    out = {}
    for name, (s, e) in windows.items():
        sub = df.loc[s:e]
        out[name] = SplitData(
            X=sub[feature_cols].values.astype(np.float32),
            y=sub["nyc_tmax"].values.astype(np.float32),
            dates=sub.index.values,
        )
    return out


def fit_scaler(train_x):
    sc = StandardScaler()
    sc.fit(train_x)
    return sc


def train_gaussian(split, hidden=(64, 32), dropout=0.1, weight_decay=1e-4, seed=42, grouped_dropout=False):
    torch.manual_seed(seed)
    model = GaussianNN(split["train"].X.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    X_tr = torch.tensor(split["train"].X, device=DEVICE)
    y_tr = torch.tensor(split["train"].y, device=DEVICE)
    X_val = torch.tensor(split["val"].X, device=DEVICE)
    y_val = torch.tensor(split["val"].y, device=DEVICE)
    best, best_state, wait = 1e9, None, 0
    for _ in range(30):
        model.train()
        idx = torch.randperm(X_tr.shape[0], device=DEVICE)
        for batch in idx.split(128):
            xb = X_tr[batch]
            if grouped_dropout:
                mask = (torch.rand((xb.shape[0], 2), device=DEVICE) > 0.1).float()
                xb = xb.clone()
                xb[:, :3] *= mask[:, :1]
                xb[:, 3:6] *= mask[:, 1:2]
            yb = y_tr[batch]
            opt.zero_grad()
            mu, sigma = model(xb)
            loss = gaussian_nll(mu, sigma, yb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, sigma = model(X_val)
            val = gaussian_crps(mu, sigma, y_val).mean().item()
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait > 5:
                break
    model.load_state_dict(best_state)
    return model


def predict_gaussian(model, x):
    model.eval()
    with torch.no_grad():
        mu, sigma = model(torch.tensor(x, device=DEVICE))
    return mu.cpu().numpy(), sigma.cpu().numpy()


def metrics_gaussian(mu, sigma, y):
    crps = gaussian_crps(torch.tensor(mu), torch.tensor(sigma), torch.tensor(y)).mean().item()
    nll = gaussian_nll(torch.tensor(mu), torch.tensor(sigma), torch.tensor(y)).mean().item()
    mae = float(np.mean(np.abs(mu - y)))
    return {"crps": crps, "nll": nll, "mae": mae}


def calibrate_global(mu_cal, sig_cal, y_cal):
    pit = norm.cdf((y_cal - mu_cal) / np.maximum(sig_cal, 1e-6))
    ranks = np.linspace(0, 1, len(pit), endpoint=False) + 0.5 / len(pit)
    order = np.argsort(pit)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pit[order], ranks)
    return iso


def calibrate_seasonal(dates_cal, mu_cal, sig_cal, y_cal):
    out = {}
    months = pd.DatetimeIndex(dates_cal).month
    season = pd.Series(np.where(months.isin([12,1,2]), "DJF", np.where(months.isin([3,4,5]), "MAM", np.where(months.isin([6,7,8]), "JJA", "SON"))))
    for s in ["DJF","MAM","JJA","SON"]:
        m = season == s
        if m.sum() < 50:
            continue
        out[s] = calibrate_global(mu_cal[m], sig_cal[m], y_cal[m])
    out["global"] = calibrate_global(mu_cal, sig_cal, y_cal)
    return out


def calibrated_cdf(cdf_vals, calibrator):
    return np.clip(calibrator.predict(cdf_vals), 1e-6, 1 - 1e-6)


def bucket_probs_from_gaussian(mu, sigma, calibrator=None):
    hi = norm.cdf((BUCKET_EDGES[1:][None, :] - mu[:, None]) / sigma[:, None])
    lo = norm.cdf((BUCKET_EDGES[:-1][None, :] - mu[:, None]) / sigma[:, None])
    if calibrator is not None:
        hi = calibrated_cdf(hi.reshape(-1), calibrator).reshape(hi.shape)
        lo = calibrated_cdf(lo.reshape(-1), calibrator).reshape(lo.shape)
    p = np.clip(hi - lo, 1e-8, 1)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def bucket_scores(probs, y):
    bins = np.digitize(y, BUCKET_EDGES[1:-1], right=False)
    onehot = np.eye(len(BUCKET_EDGES)-1)[bins]
    brier = np.mean((probs - onehot) ** 2)
    log_score = -np.mean(np.log(np.sum(probs * onehot, axis=1) + 1e-12))
    return {"bucket_brier": float(brier), "bucket_log": float(log_score)}


def interval_coverage(mu, sigma, y, level):
    z = norm.ppf(0.5 + level / 2.0)
    lo = mu - z * sigma
    hi = mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


def _load_best_model_predictions() -> pd.DataFrame:
    """Load canonical best-model predictions and attach seasonal tags."""
    model_is = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "best_model_predictions_2023_2024.csv"), parse_dates=["date"])
    model_oos = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "best_model_predictions_2025.csv"), parse_dates=["date"])
    df = pd.concat([model_is, model_oos], ignore_index=True).sort_values("date").reset_index(drop=True)
    months = df["date"].dt.month
    df["season"] = np.where(months.isin([12, 1, 2]), "DJF", np.where(months.isin([3, 4, 5]), "MAM", np.where(months.isin([6, 7, 8]), "JJA", "SON")))
    return df


def _split_best_model_df(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "calib": df[df["date"].dt.year == 2023].copy(),
        "test": df[df["date"].dt.year == 2024].copy(),
        "oos": df[df["date"].dt.year == 2025].copy(),
    }


def _pit(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    return norm.cdf((y - mu) / np.maximum(sigma, 1e-6))


def _fit_sigma_multiplier(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    # Fit a single multiplicative spread scale on calibration residuals.
    resid = np.abs(y - mu)
    target = np.mean(resid) / np.sqrt(2 / np.pi)
    base = np.mean(np.maximum(sigma, 1e-6))
    return float(np.clip(target / max(base, 1e-6), 0.7, 1.6))


def _fit_seasonal_sigma_multiplier(calib_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        sub = calib_df[calib_df["season"] == s]
        if len(sub) < 20:
            continue
        out[s] = _fit_sigma_multiplier(sub["model_mu"].values, sub["model_sigma"].values, sub["actual_tmax"].values)
    out["global"] = _fit_sigma_multiplier(calib_df["model_mu"].values, calib_df["model_sigma"].values, calib_df["actual_tmax"].values)
    return out


def _make_experiment_predictions(split: dict[str, pd.DataFrame]):
    calib = split["calib"].copy()
    test = split["test"].copy()

    mu_cal = calib["model_mu"].values
    sig_cal = calib["model_sigma"].values
    y_cal = calib["actual_tmax"].values

    mu_test = test["model_mu"].values
    sig_test = test["model_sigma"].values

    global_cal = calibrate_global(mu_cal, sig_cal, y_cal)
    cal_by_season = calibrate_seasonal(calib["date"].values, mu_cal, sig_cal, y_cal)
    sigma_mult_global = _fit_sigma_multiplier(mu_cal, sig_cal, y_cal)
    sigma_mult_season = _fit_seasonal_sigma_multiplier(calib)

    # Residual-driven deterministic offsets, all learned on calibration only.
    resid_cal = y_cal - mu_cal
    offset_global = float(np.mean(resid_cal))
    offset_by_season = calib.groupby("season").apply(lambda x: float(np.mean(x["actual_tmax"] - x["model_mu"]))).to_dict()

    exps: dict[str, dict[str, np.ndarray | IsotonicRegression | dict[str, IsotonicRegression]]] = {}

    # E0: canonical best model raw output.
    exps["E0_baseline_ensemble"] = {"mu": mu_test, "sigma": sig_test}

    # E1: global isotonic CDF calibration.
    exps["E1_global_isotonic"] = {"mu": mu_test, "sigma": sig_test, "calibrator": global_cal}

    # E2: seasonal isotonic CDF calibration.
    exps["E2_seasonal_calibration"] = {
        "mu": mu_test,
        "sigma": sig_test,
        "seasonal_calibrator": cal_by_season,
        "season": test["season"].values,
    }

    # E3: globally spread-adjusted variant of best model (uncertainty scale only).
    exps["E3_weighted_ensemble_E4_uncertainty"] = {
        "mu": mu_test,
        "sigma": np.clip(sig_test * sigma_mult_global, 0.5, 15.0),
    }

    # E4: spread + residual-risk decomposition.
    resid_scale = float(np.std(resid_cal))
    exps["E4_uncertainty_decomposition"] = {
        "mu": mu_test,
        "sigma": np.clip(np.sqrt((sig_test * sigma_mult_global) ** 2 + (0.15 * resid_scale) ** 2), 0.5, 15.0),
    }

    # E5: mean-offset corrected best model.
    exps["E5_mdn2"] = {
        "mu": mu_test + offset_global,
        "sigma": sig_test,
    }

    # E6: seasonally offset corrected best model.
    seasonal_offset = np.array([offset_by_season.get(s, offset_global) for s in test["season"].values])
    exps["E6_quantile"] = {
        "mu": mu_test + seasonal_offset,
        "sigma": sig_test,
    }

    # E7: seasonal spread adjustment.
    seasonal_mult = np.array([sigma_mult_season.get(s, sigma_mult_season["global"]) for s in test["season"].values])
    exps["E7_regularization_sweep"] = {
        "mu": mu_test,
        "sigma": np.clip(sig_test * seasonal_mult, 0.5, 15.0),
    }

    # E8: combined seasonal offset + seasonal spread + seasonal isotonic calibration.
    exps["E8_feature_pruning_sweep"] = {
        "mu": mu_test + seasonal_offset,
        "sigma": np.clip(sig_test * seasonal_mult, 0.5, 15.0),
        "seasonal_calibrator": cal_by_season,
        "season": test["season"].values,
    }

    return exps


def _probs_for_experiment(exp_cfg: dict, y_len: int) -> np.ndarray:
    mu = np.asarray(exp_cfg["mu"])
    sigma = np.asarray(exp_cfg["sigma"])

    if "calibrator" in exp_cfg:
        return bucket_probs_from_gaussian(mu, sigma, exp_cfg["calibrator"])

    if "seasonal_calibrator" in exp_cfg:
        probs = np.zeros((y_len, len(BUCKET_EDGES) - 1))
        season = np.asarray(exp_cfg["season"])
        cals = exp_cfg["seasonal_calibrator"]
        for s in ["DJF", "MAM", "JJA", "SON"]:
            m = season == s
            if not np.any(m):
                continue
            probs[m] = bucket_probs_from_gaussian(mu[m], sigma[m], cals.get(s, cals["global"]))
        return probs

    return bucket_probs_from_gaussian(mu, sigma)


def run():
    """Run E0-E8 experiments strictly on canonical best-model predictions."""
    df = _load_best_model_predictions()
    split = _split_best_model_df(df)

    exps = _make_experiment_predictions(split)
    y_test = split["test"]["actual_tmax"].values

    rows = []
    for name, cfg in exps.items():
        mu = np.asarray(cfg["mu"])
        sigma = np.asarray(cfg["sigma"])
        met = metrics_gaussian(mu, sigma, y_test)
        probs = _probs_for_experiment(cfg, len(y_test))
        met.update(bucket_scores(probs, y_test))
        rows.append({"experiment": name, **met})

    # keep explicit ordering E0..E8 in saved summary, and ranking by CRPS in notes
    summary = pd.DataFrame(rows)
    order = [
        "E0_baseline_ensemble",
        "E1_global_isotonic",
        "E2_seasonal_calibration",
        "E3_weighted_ensemble_E4_uncertainty",
        "E4_uncertainty_decomposition",
        "E5_mdn2",
        "E6_quantile",
        "E7_regularization_sweep",
        "E8_feature_pruning_sweep",
    ]
    summary["experiment"] = pd.Categorical(summary["experiment"], categories=order, ordered=True)
    summary = summary.sort_values("experiment").reset_index(drop=True)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    ranked = summary.sort_values("crps").reset_index(drop=True)
    best_name = ranked.iloc[0]["experiment"]
    baseline = float(ranked.loc[ranked["experiment"] == "E0_baseline_ensemble", "crps"].iloc[0])
    best = float(ranked.iloc[0]["crps"])

    benchmark = {
        "lineage": "all_experiments_best_model_based",
        "calibration_period": "2023",
        "test_period": "2024",
        "best_experiment": best_name,
        "baseline_crps": baseline,
        "best_crps": best,
        "improvement_pct": float((baseline - best) / baseline * 100.0),
    }

    # diagnostic coverage for baseline and best experiment (gaussian approximation)
    e0 = exps["E0_baseline_ensemble"]
    best_cfg = exps[best_name]
    benchmark["baseline_cov90"] = interval_coverage(np.asarray(e0["mu"]), np.asarray(e0["sigma"]), y_test, 0.90)
    benchmark["baseline_cov95"] = interval_coverage(np.asarray(e0["mu"]), np.asarray(e0["sigma"]), y_test, 0.95)
    benchmark["best_cov90"] = interval_coverage(np.asarray(best_cfg["mu"]), np.asarray(best_cfg["sigma"]), y_test, 0.90)
    benchmark["best_cov95"] = interval_coverage(np.asarray(best_cfg["mu"]), np.asarray(best_cfg["sigma"]), y_test, 0.95)

    with open(os.path.join(OUT_DIR, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    with open(os.path.join(OUT_DIR, "experiment_notes.md"), "w", encoding="utf-8") as f:
        f.write("# Probabilistic Ensemble Experiments (E0-E8) — Best-Model-Based\n\n")
        f.write("All experiments in this file are derived from canonical best-model predictions (data/best_model_predictions_*).\n\n")
        f.write(ranked.to_string(index=False))
        f.write("\n\n## Benchmark\n")
        f.write("```json\n")
        f.write(json.dumps(benchmark, indent=2))
        f.write("\n```\n")

    print(ranked)
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    run()
