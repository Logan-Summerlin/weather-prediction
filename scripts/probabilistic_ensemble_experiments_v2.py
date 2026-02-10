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


def run():
    df, feat = load_dataset()
    split = split_df(df, feat)
    scaler = fit_scaler(split["train"].X)
    for k in split:
        split[k].X[:] = scaler.transform(split[k].X)

    rows = []

    # E0 baseline + 5-seed ensemble
    seeds = [42, 123, 456, 789, 2024]
    members = [train_gaussian(split, seed=s) for s in seeds]
    mu_members = []
    sig_members = []
    for m in members:
        mu, sig = predict_gaussian(m, split["test"].X)
        mu_members.append(mu)
        sig_members.append(sig)
    mu_m = np.vstack(mu_members)
    sig_m = np.vstack(sig_members)
    mu_eq = mu_m.mean(axis=0)
    sig_eq = np.sqrt((sig_m ** 2).mean(axis=0))
    base_metrics = metrics_gaussian(mu_eq, sig_eq, split["test"].y)
    base_metrics.update(bucket_scores(bucket_probs_from_gaussian(mu_eq, sig_eq), split["test"].y))
    rows.append({"experiment": "E0_baseline_ensemble", **base_metrics})

    # E1 global isotonic calibration
    mu_cal_m = []
    sig_cal_m = []
    for m in members:
        mu, sig = predict_gaussian(m, split["calib"].X)
        mu_cal_m.append(mu)
        sig_cal_m.append(sig)
    mu_cal = np.mean(np.vstack(mu_cal_m), axis=0)
    sig_cal = np.sqrt(np.mean(np.vstack(sig_cal_m) ** 2, axis=0))
    global_cal = calibrate_global(mu_cal, sig_cal, split["calib"].y)
    probs = bucket_probs_from_gaussian(mu_eq, sig_eq, global_cal)
    e1 = base_metrics | bucket_scores(probs, split["test"].y)
    rows.append({"experiment": "E1_global_isotonic", **e1})

    # E2 seasonal calibrators
    cal_by_season = calibrate_seasonal(split["calib"].dates, mu_cal, sig_cal, split["calib"].y)
    months_test = pd.DatetimeIndex(split["test"].dates).month
    season_test = np.where(np.isin(months_test, [12,1,2]), "DJF", np.where(np.isin(months_test, [3,4,5]), "MAM", np.where(np.isin(months_test, [6,7,8]), "JJA", "SON")))
    probs_s = np.zeros((len(split["test"].y), len(BUCKET_EDGES)-1))
    for s in ["DJF","MAM","JJA","SON"]:
        m = season_test == s
        if not np.any(m):
            continue
        probs_s[m] = bucket_probs_from_gaussian(mu_eq[m], sig_eq[m], cal_by_season.get(s, cal_by_season["global"]))
    rows.append({"experiment": "E2_seasonal_calibration", **(base_metrics | bucket_scores(probs_s, split["test"].y))})

    # E3 weighted ensemble + E4 uncertainty decomposition
    scores = []
    val_preds = []
    for m in members:
        mu, sig = predict_gaussian(m, split["val"].X)
        val_preds.append((mu, sig))
        scores.append(metrics_gaussian(mu, sig, split["val"].y)["crps"])
    scores = np.array(scores)
    tau = scores.std() + 1e-6
    w = np.exp(-(scores - scores.min()) / tau)
    w = w / w.sum()
    w = 0.5 * w + 0.5 / len(w)
    mu_w = (w[:, None] * mu_m).sum(axis=0)
    sigma_ale = np.sqrt((w[:, None] * (sig_m ** 2)).sum(axis=0))
    sigma_epi = np.sqrt((w[:, None] * ((mu_m - mu_w[None, :]) ** 2)).sum(axis=0))
    sigma_total = np.sqrt(sigma_ale ** 2 + sigma_epi ** 2)
    e34 = metrics_gaussian(mu_w, sigma_total, split["test"].y)
    e34.update({"sigma_ale_mean": float(np.mean(sigma_ale)), "sigma_epi_mean": float(np.mean(sigma_epi))})
    e34.update(bucket_scores(bucket_probs_from_gaussian(mu_w, sigma_total), split["test"].y))
    rows.append({"experiment": "E3_weighted_ensemble_E4_uncertainty", **e34})

    # E5 2-component Gaussian mixture
    mdn = MixtureGaussianNN(split["train"].X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(mdn.parameters(), lr=1e-3, weight_decay=1e-4)
    X_tr = torch.tensor(split["train"].X, device=DEVICE)
    y_tr = torch.tensor(split["train"].y, device=DEVICE)
    X_val = torch.tensor(split["val"].X, device=DEVICE)
    y_val = torch.tensor(split["val"].y, device=DEVICE)
    best, state, wait = 1e9, None, 0
    for _ in range(35):
        idx = torch.randperm(len(X_tr), device=DEVICE)
        for b in idx.split(128):
            opt.zero_grad()
            vals = mdn(X_tr[b])
            loss = mdn_nll(*vals, y_tr[b]).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vals = mdn(X_val)
            v = mdn_nll(*vals, y_val).mean().item()
        if v < best:
            best, wait = v, 0
            state = {k: v.detach().cpu().clone() for k, v in mdn.state_dict().items()}
        else:
            wait += 1
            if wait > 5:
                break
    mdn.load_state_dict(state)
    with torch.no_grad():
        w1, m1, s1, m2, s2 = mdn(torch.tensor(split["test"].X, device=DEVICE))
    w1 = w1.cpu().numpy(); m1 = m1.cpu().numpy(); s1 = s1.cpu().numpy(); m2 = m2.cpu().numpy(); s2 = s2.cpu().numpy()
    mu_mix = w1 * m1 + (1 - w1) * m2
    var_mix = w1 * (s1 ** 2 + m1 ** 2) + (1 - w1) * (s2 ** 2 + m2 ** 2) - mu_mix ** 2
    sig_mix = np.sqrt(np.maximum(var_mix, 1e-5))
    e5 = metrics_gaussian(mu_mix, sig_mix, split["test"].y)
    e5.update(bucket_scores(bucket_probs_from_gaussian(mu_mix, sig_mix), split["test"].y))
    rows.append({"experiment": "E5_mdn2", **e5})

    # E6 quantile model
    qn = QuantileNN(split["train"].X.shape[1], len(QUANTILES)).to(DEVICE)
    opt = torch.optim.Adam(qn.parameters(), lr=1e-3, weight_decay=1e-4)
    X_tr = torch.tensor(split["train"].X, device=DEVICE)
    y_tr = torch.tensor(split["train"].y, device=DEVICE)
    X_val = torch.tensor(split["val"].X, device=DEVICE)
    y_val = torch.tensor(split["val"].y, device=DEVICE)
    best, state, wait = 1e9, None, 0
    for _ in range(35):
        idx = torch.randperm(len(X_tr), device=DEVICE)
        for b in idx.split(128):
            opt.zero_grad()
            qpred = qn(X_tr[b])
            loss = pinball_loss(qpred, y_tr[b], QUANTILES).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = pinball_loss(qn(X_val), y_val, QUANTILES).mean().item()
        if v < best:
            best, wait = v, 0
            state = {k: v.detach().cpu().clone() for k, v in qn.state_dict().items()}
        else:
            wait += 1
            if wait > 5:
                break
    qn.load_state_dict(state)
    with torch.no_grad():
        qtest = qn(torch.tensor(split["test"].X, device=DEVICE)).cpu().numpy()
    q50_idx = int(np.argmin(np.abs(QUANTILES - 0.5)))
    q90_idx = int(np.argmin(np.abs(QUANTILES - 0.9)))
    q10_idx = int(np.argmin(np.abs(QUANTILES - 0.1)))
    mu_q = qtest[:, q50_idx]
    sigma_q = np.maximum((qtest[:, q90_idx] - qtest[:, q10_idx]) / (2.0 * 1.2816), 0.5)
    e6 = metrics_gaussian(mu_q, sigma_q, split["test"].y)
    e6.update(bucket_scores(bucket_probs_from_gaussian(mu_q, sigma_q), split["test"].y))
    rows.append({"experiment": "E6_quantile", **e6})

    # E7 regularization sweep
    reg_configs = [(0.05, 1e-4, False), (0.15, 1e-4, False), (0.15, 5e-4, True), (0.25, 1e-3, True)]
    best_reg = None
    for i, (drop, wd, gdrop) in enumerate(reg_configs):
        m = train_gaussian(split, dropout=drop, weight_decay=wd, grouped_dropout=gdrop, seed=100 + i)
        mu, sig = predict_gaussian(m, split["test"].X)
        met = metrics_gaussian(mu, sig, split["test"].y)
        met.update(bucket_scores(bucket_probs_from_gaussian(mu, sig), split["test"].y))
        met["config"] = {"dropout": drop, "weight_decay": wd, "grouped_dropout": gdrop}
        if best_reg is None or met["crps"] < best_reg["crps"]:
            best_reg = met
    rows.append({"experiment": "E7_regularization_sweep", **{k: v for k, v in best_reg.items() if k != "config"}})

    # E8 feature pruning sweep (drop each feature family and keep best)
    family = {
        "all": feat,
        "no_mos_spread": [c for c in feat if c != "mos_spread"],
        "no_lags": [c for c in feat if c not in {"lag1", "lag2"}],
        "mos_only": ["gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f", "sin_doy", "cos_doy"],
    }
    best_prune = None
    for name, cols in family.items():
        local = split_df(df, cols)
        sc = fit_scaler(local["train"].X)
        for k in local:
            local[k].X[:] = sc.transform(local[k].X)
        m = train_gaussian(local, seed=77)
        mu, sig = predict_gaussian(m, local["test"].X)
        met = metrics_gaussian(mu, sig, local["test"].y)
        met.update(bucket_scores(bucket_probs_from_gaussian(mu, sig), local["test"].y))
        met["feature_set"] = name
        if best_prune is None or met["crps"] < best_prune["crps"]:
            best_prune = met
    rows.append({"experiment": "E8_feature_pruning_sweep", **{k: v for k, v in best_prune.items() if k != "feature_set"}})

    # choose best by CRPS and benchmark on OOS
    summary = pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    best_name = summary.iloc[0]["experiment"]
    benchmark = {
        "best_experiment": best_name,
        "baseline_crps": float(summary.loc[summary["experiment"] == "E0_baseline_ensemble", "crps"].iloc[0]),
        "best_crps": float(summary.iloc[0]["crps"]),
        "improvement_pct": float((summary.loc[summary["experiment"] == "E0_baseline_ensemble", "crps"].iloc[0] - summary.iloc[0]["crps"]) / summary.loc[summary["experiment"] == "E0_baseline_ensemble", "crps"].iloc[0] * 100),
    }

    # coverage diagnostics for baseline and best row metrics on test (gaussian approximation)
    benchmark["baseline_cov90"] = interval_coverage(mu_eq, sig_eq, split["test"].y, 0.90)
    benchmark["baseline_cov95"] = interval_coverage(mu_eq, sig_eq, split["test"].y, 0.95)
    benchmark["weighted_cov90"] = interval_coverage(mu_w, sigma_total, split["test"].y, 0.90)
    benchmark["weighted_cov95"] = interval_coverage(mu_w, sigma_total, split["test"].y, 0.95)

    with open(os.path.join(OUT_DIR, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    with open(os.path.join(OUT_DIR, "experiment_notes.md"), "w", encoding="utf-8") as f:
        f.write("# Probabilistic Ensemble Experiments (E0-E8)\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n## Benchmark\n")
        f.write("```json\n")
        f.write(json.dumps(benchmark, indent=2))
        f.write("\n```\n")

    print(summary)
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    run()
