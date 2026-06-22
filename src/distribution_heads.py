"""Distribution heads: Gaussian, 7-quantile, and 2-component mixture.

Phase 2 deliverable #6.  Provides three forecast representations and the
scoring primitives (OOS CRPS + contract Brier) used to choose the best head
per city.  The torch heads share a single training loop; the numpy scoring
functions are framework-free so they can be unit-tested without torch and
reused by the diagnostics / hparam scripts.

Representations
---------------
* **gaussian** — ``(mu, sigma)``.  Scored with the closed-form Gaussian CRPS
  (via :func:`src.calibration.compute_crps`); priced with
  :func:`src.bucket_semantics.bucket_prob_from_edges`.
* **quantile** — monotone quantile values at fixed levels (default the 7
  levels 0.05..0.95).  CRPS estimated from the average pinball loss; priced by
  interpolating the implied CDF (settlement-rounding-aware).
* **mixture** — a K-component Gaussian mixture ``(weights, mus, sigmas)``.
  Scored with the *exact* Grimit et al. (2006) closed-form mixture CRPS;
  priced with :func:`src.bucket_semantics.mixture_bucket_prob_from_edges`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# NOTE: the Gaussian and mixture heads price contracts through
# src.bucket_semantics at their call sites (e.g. the comparison script).  The
# quantile head applies the same documented -0.5 settlement-rounding shift
# locally because bucket_semantics has no quantile-function entry point.

DEFAULT_QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Closed-form Gaussian / mixture CRPS (numpy)
# ---------------------------------------------------------------------------
def _crps_unit(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """A(m, s) = m*(2*Phi(m/s) - 1) + 2*s*phi(m/s); the CRPS building block."""
    s = np.maximum(s, 1e-12)
    z = m / s
    return m * (2.0 * norm.cdf(z) - 1.0) + 2.0 * s * norm.pdf(z)


def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample closed-form Gaussian CRPS.

    ``CRPS = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]`` with
    ``z = (y - mu) / sigma``.  Equivalent to the single-component reduction of
    :func:`gaussian_mixture_crps`.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    y = np.asarray(y, dtype=float)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - _INV_SQRT_PI)


def gaussian_mixture_crps(
    weights: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Exact CRPS for a Gaussian mixture (Grimit et al. 2006).

    ``weights`` shape ``(K,)`` or ``(n, K)``; ``mus``/``sigmas`` shape
    ``(n, K)``; ``y`` shape ``(n,)``.  Reduces to :func:`gaussian_crps` for a
    single component.
    """
    mus = np.atleast_2d(np.asarray(mus, dtype=float))
    sigmas = np.maximum(np.atleast_2d(np.asarray(sigmas, dtype=float)), 1e-12)
    n, k = mus.shape
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = np.broadcast_to(w, (n, k)).copy()
    w = w / w.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=float).reshape(n, 1)

    term1 = np.sum(w * _crps_unit(y - mus, sigmas), axis=1)

    term2 = np.zeros(n)
    for i in range(k):
        for j in range(k):
            m = mus[:, i] - mus[:, j]
            s = np.sqrt(sigmas[:, i] ** 2 + sigmas[:, j] ** 2)
            term2 += w[:, i] * w[:, j] * _crps_unit(m, s)
    return term1 - 0.5 * term2


# ---------------------------------------------------------------------------
# Quantile representation
# ---------------------------------------------------------------------------
def pinball_loss(levels: np.ndarray, qvals: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean pinball (quantile) loss per sample over the given levels.

    ``levels`` shape ``(Q,)``; ``qvals`` shape ``(n, Q)``; ``y`` shape ``(n,)``.
    """
    levels = np.asarray(levels, dtype=float)
    qvals = np.asarray(qvals, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    err = y - qvals
    loss = np.maximum(levels * err, (levels - 1.0) * err)
    return loss.mean(axis=1)


def quantile_crps(levels: np.ndarray, qvals: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS estimate from quantiles: ``2 * mean_tau pinball_tau``.

    For evenly spread levels covering (0,1) this is the standard quantile-based
    CRPS approximation (CRPS = 2 * integral of pinball over tau).
    """
    return 2.0 * pinball_loss(levels, qvals, y)


def quantile_cdf_at(levels: np.ndarray, qvals: np.ndarray, x: float) -> np.ndarray:
    """Implied CDF value F(x) per sample by interpolating the quantile function.

    Linearly interpolates ``x`` within each row's (sorted) quantile values,
    mapping back to probability levels; clamps outside the support to [0, 1].
    """
    levels = np.asarray(levels, dtype=float)
    qvals = np.asarray(qvals, dtype=float)
    n = qvals.shape[0]
    out = np.empty(n)
    for i in range(n):
        row = np.sort(qvals[i])
        out[i] = float(np.interp(x, row, levels, left=0.0, right=1.0))
    return out


def quantile_bucket_prob_from_edges(
    levels: np.ndarray, qvals: np.ndarray, lo: float, hi: float, open_edge: float = 900.0
) -> np.ndarray:
    """Bucket probability ``P(round(T) in [lo, hi))`` from quantile forecasts.

    Applies the same -0.5 settlement-rounding shift as the Gaussian path so the
    quantile head stays contract-faithful.
    """
    lo_x, hi_x = lo - 0.5, hi - 0.5
    cdf_hi = 1.0 if hi >= open_edge else quantile_cdf_at(levels, qvals, hi_x)
    cdf_lo = 0.0 if lo <= -open_edge else quantile_cdf_at(levels, qvals, lo_x)
    return np.clip(cdf_hi - cdf_lo, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Unified scoring across heads
# ---------------------------------------------------------------------------
@dataclass
class HeadScore:
    head: str
    mean_crps: float
    contract_brier: float
    n: int


def select_best_head(scores: list[HeadScore]) -> str:
    """Best head by mean OOS CRPS (the primary probabilistic score)."""
    return min(scores, key=lambda s: s.mean_crps).head


# ---------------------------------------------------------------------------
# Torch heads (imported lazily so numpy scoring works without torch)
# ---------------------------------------------------------------------------
HEAD_FAMILIES = ("gaussian", "quantile", "mixture")


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("torch is required for the neural distribution heads") from exc


def _mlp_body(n_features: int, hidden: int, depth: int):
    """A ReLU MLP trunk with ``depth`` hidden layers."""
    import torch.nn as nn
    layers, in_dim = [], n_features
    for _ in range(max(1, depth)):
        layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
        in_dim = hidden
    return nn.Sequential(*layers)


def build_gaussian_net(n_features: int, hidden=64, depth=2):
    """An MLP emitting heteroscedastic Gaussian ``(mu, sigma)``.

    log_sigma is clamped to avoid the variance-collapse pathology seen in
    Phase 0 (Austin's blown-up constant sigma).  The caller centers the target,
    so the mu head starts near the mean.
    """
    _require_torch()
    import torch
    import torch.nn as nn

    class GaussianNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = _mlp_body(n_features, hidden, depth)
            self.mu = nn.Linear(hidden, 1)
            self.log_sigma = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.body(x)
            mu = self.mu(h).squeeze(-1)
            sigma = torch.clamp(self.log_sigma(h).squeeze(-1), -2.0, 4.0).exp()
            return mu, sigma

    return GaussianNet()


def gaussian_nll_torch(mu, sigma, y):
    """Mean Gaussian negative log-likelihood (torch) for the Gaussian head."""
    _require_torch()
    import torch
    return (
        0.5 * math.log(2 * math.pi)
        + torch.log(sigma)
        + 0.5 * ((y - mu) / sigma) ** 2
    ).mean()


def build_quantile_net(n_features: int, levels=DEFAULT_QUANTILE_LEVELS, hidden=64, depth=2):
    """An MLP emitting monotone quantiles via a cumulative-softplus head."""
    _require_torch()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class QuantileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.levels = tuple(levels)
            self.body = _mlp_body(n_features, hidden, depth)
            self.base = nn.Linear(hidden, 1)
            self.deltas = nn.Linear(hidden, len(levels) - 1)

        def forward(self, x):
            h = self.body(x)
            base = self.base(h)
            steps = F.softplus(self.deltas(h))  # non-negative => monotone
            return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)

    return QuantileNet()


def build_mixture_net(n_features: int, n_components: int = 2, hidden=64, depth=2):
    """A K-component Gaussian mixture density network (weights, mus, sigmas)."""
    _require_torch()
    import torch
    import torch.nn as nn

    class MixtureDensityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = n_components
            self.body = _mlp_body(n_features, hidden, depth)
            self.logits = nn.Linear(hidden, n_components)
            self.mu = nn.Linear(hidden, n_components)
            self.log_sigma = nn.Linear(hidden, n_components)

        def forward(self, x):
            h = self.body(x)
            w = torch.softmax(self.logits(h), dim=1)
            mu = self.mu(h)
            sigma = torch.clamp(self.log_sigma(h), -2.0, 4.0).exp()
            return w, mu, sigma

    return MixtureDensityNet()


def pinball_loss_torch(levels, preds, y):
    """Mean pinball loss (torch) for training the quantile head."""
    _require_torch()
    import torch
    lv = torch.as_tensor(levels, dtype=preds.dtype, device=preds.device).view(1, -1)
    err = y.view(-1, 1) - preds
    return torch.maximum(lv * err, (lv - 1.0) * err).mean()


def mixture_nll_torch(w, mu, sigma, y):
    """Mean negative log-likelihood (torch) for training the mixture head."""
    _require_torch()
    import torch
    y = y.view(-1, 1)
    log_comp = (
        -0.5 * math.log(2 * math.pi)
        - torch.log(sigma)
        - 0.5 * ((y - mu) / sigma) ** 2
    )
    log_prob = torch.logsumexp(torch.log(w + 1e-12) + log_comp, dim=1)
    return -log_prob.mean()


# ---------------------------------------------------------------------------
# Unified head registry: build / loss / predict / score
# ---------------------------------------------------------------------------
def build_head(family: str, n_features: int, levels=DEFAULT_QUANTILE_LEVELS,
               hidden=64, depth=2, n_components=2):
    """Construct a head net for ``family`` in {gaussian, quantile, mixture}."""
    if family == "gaussian":
        return build_gaussian_net(n_features, hidden=hidden, depth=depth)
    if family == "quantile":
        return build_quantile_net(n_features, levels=levels, hidden=hidden, depth=depth)
    if family == "mixture":
        return build_mixture_net(n_features, n_components=n_components, hidden=hidden, depth=depth)
    raise ValueError(f"Unknown head family {family!r}; expected one of {HEAD_FAMILIES}")


def head_loss_fn(family: str, levels=DEFAULT_QUANTILE_LEVELS):
    """Return ``loss(net_output, y_tensor)`` for the family."""
    if family == "gaussian":
        return lambda out, y: gaussian_nll_torch(out[0], out[1], y)
    if family == "quantile":
        return lambda out, y: pinball_loss_torch(levels, out, y)
    if family == "mixture":
        return lambda out, y: mixture_nll_torch(out[0], out[1], out[2], y)
    raise ValueError(f"Unknown head family {family!r}")


def train_head_net(net, loss_fn, X_tr, y_tr, X_va, y_va, epochs=150, lr=1e-2,
                   weight_decay=0.0, batch_size=256, patience=40, seed=0):
    """Mini-batch Adam with ReduceLROnPlateau and val-loss early stopping.

    Tensors are expected as float32 torch tensors.  Returns the net with the
    best-validation weights restored.
    """
    _require_torch()
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
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
            best_val = val
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val


def predict_head(family: str, net, X_tensor, y_mean: float,
                 levels=DEFAULT_QUANTILE_LEVELS) -> dict:
    """Run a trained head and return numpy forecast params (de-centered).

    Keys by family: gaussian -> {mu, sigma}; quantile -> {levels, qvals};
    mixture -> {weights, mus, sigmas}.
    """
    _require_torch()
    import torch
    net.eval()
    with torch.no_grad():
        out = net(X_tensor)
    if family == "gaussian":
        mu, sigma = out
        return {"mu": mu.numpy() + y_mean, "sigma": np.maximum(sigma.numpy(), 1e-3)}
    if family == "quantile":
        return {"levels": np.array(levels), "qvals": out.numpy() + y_mean}
    if family == "mixture":
        w, mu, sigma = out
        return {"weights": w.numpy(), "mus": mu.numpy() + y_mean,
                "sigmas": np.maximum(sigma.numpy(), 1e-3)}
    raise ValueError(f"Unknown head family {family!r}")


def head_crps(family: str, pred: dict, y: np.ndarray) -> float:
    """Mean OOS CRPS for a head's prediction dict (from :func:`predict_head`)."""
    if family == "gaussian":
        return float(np.mean(gaussian_crps(pred["mu"], pred["sigma"], y)))
    if family == "quantile":
        return float(np.mean(quantile_crps(pred["levels"], pred["qvals"], y)))
    if family == "mixture":
        return float(np.mean(gaussian_mixture_crps(pred["weights"], pred["mus"], pred["sigmas"], y)))
    raise ValueError(f"Unknown head family {family!r}")
