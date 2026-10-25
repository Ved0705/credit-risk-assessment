"""Evaluation helpers. Positive class (1) = HIGH RISK (bad credit)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

# Cost matrix published with the German Credit dataset:
# classifying a BAD applicant as GOOD (false negative) costs 5,
# classifying a GOOD applicant as BAD (false positive) costs 1.
COST_FN = 5
COST_FP = 1


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "misclassification_cost": int(COST_FN * fn + COST_FP * fp),
    }


def best_cost_threshold(y_true, proba) -> float:
    """Threshold that minimises 5*FN + 1*FP on (out-of-fold) training predictions."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best_t, best_cost = 0.5, float("inf")
    for t in np.arange(0.05, 0.951, 0.01):
        pred = proba >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        cost = COST_FN * fn + COST_FP * fp
        if cost < best_cost:
            best_t, best_cost = float(t), cost
    return round(best_t, 2)
