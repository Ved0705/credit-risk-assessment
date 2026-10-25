"""
Bootstrap confidence intervals.

The held-out test set has only 200 applicants, so every metric computed on it is
uncertain. We resample the test set with replacement (B = 1000 times, fixed seed),
recompute the metric on each resample and report the 2.5th and 97.5th percentiles
(percentile bootstrap, 95 % interval). The model is NOT retrained: the interval
describes test-set sampling uncertainty only.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .evaluate import COST_FN, COST_FP

N_BOOT = 1000
SEED = 42


def _percentile_ci(values) -> list[float] | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else np.nan


def metric_cis(y_true, proba, threshold: float, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95 % bootstrap intervals for the headline classification metrics."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "misclassification_cost")}
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt, pt = y[i], p[i]
        pred = pt >= threshold
        tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
        fn = int((~pred & (yt == 1)).sum()); tn = int((~pred & (yt == 0)).sum())
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        out["accuracy"].append((tp + tn) / n)
        out["precision"].append(prec)
        out["recall"].append(rec)
        out["f1"].append(_safe_div(2 * prec * rec, prec + rec) if np.isfinite(prec) and np.isfinite(rec) else np.nan)
        out["roc_auc"].append(roc_auc_score(yt, pt) if 0 < yt.sum() < n else np.nan)
        out["misclassification_cost"].append(COST_FN * fn + COST_FP * fp)
    return {k: _percentile_ci(v) for k, v in out.items()}


def _group_gaps(y, pred, groups, labels) -> tuple[float, float]:
    """Demographic parity difference and equalized odds difference (same definitions as Fairlearn)."""
    sel, tpr, fpr = [], [], []
    for g in labels:
        m = groups == g
        if not m.any():
            return np.nan, np.nan
        sel.append(pred[m].mean())
        pos, neg = m & (y == 1), m & (y == 0)
        tpr.append(pred[pos].mean() if pos.any() else np.nan)
        fpr.append(pred[neg].mean() if neg.any() else np.nan)
    dpd = max(sel) - min(sel)
    eod = max(np.ptp(tpr), np.ptp(fpr))  # nan if any group lacks positives or negatives
    return dpd, eod


def fairness_gap_cis(y_true, y_pred, groups, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95 % bootstrap intervals for demographic parity difference and equalized odds difference.

    Resamples where a group is missing (or has no positives / negatives) are skipped;
    `valid_resamples` reports how many were usable.
    """
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(bool)
    g = np.asarray(groups)
    labels = sorted(set(g.tolist()))
    rng = np.random.default_rng(seed)
    dpds, eods = [], []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d, e = _group_gaps(y[i], pred[i], g[i], labels)
        dpds.append(d); eods.append(e)
    return {
        "demographic_parity_difference": _percentile_ci(dpds),
        "equalized_odds_difference": _percentile_ci(eods),
        "valid_resamples": int(np.isfinite(eods).sum()),
        "n_boot": n_boot,
    }
