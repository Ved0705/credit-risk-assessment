"""
Group fairness audit on the held-out TEST set using Fairlearn.

Positive class = HIGH RISK (rejection). In plain language:
  * approval rate        = share of the group predicted LOW RISK
  * false positive rate  = good payers wrongly flagged HIGH RISK (unfair rejections)
  * false negative rate  = bad payers missed (flagged LOW RISK)
  * demographic parity difference = largest gap in approval rate between groups
  * equalized odds difference     = largest gap in FPR or TPR between groups

Each gap comes with a 95 % bootstrap interval (test-set resampling). An attribute in
which any group has fewer than MIN_GROUP_SIZE test applicants gets the status
INSUFFICIENT DATA instead of a verdict, and is left out of the overall status.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from fairlearn.metrics import (
    MetricFrame, count, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, false_negative_rate, false_positive_rate, selection_rate,
)
from sklearn.metrics import accuracy_score

from ..ml.model_loader import get_model
from ..ml.uncertainty import fairness_gap_cis
from ..ml.preprocessing import MODEL_FEATURES, TARGET

ATTRIBUTES = {
    "sex": "Sex, derived from the dataset's 'personal status and sex' field. NOT used by the model.",
    "age_group": "Age under 25 vs 25 and over. Age IS a model input, so differences can be direct.",
    "foreign_worker_group": "Foreign-worker status. NOT used by the model.",
}
ACCEPTABLE, ATTENTION = 0.10, 0.20
MIN_GROUP_SIZE = 30
INSUFFICIENT = "INSUFFICIENT DATA"

LIMITATIONS = [
    "The test set has only 200 applicants, so group metrics have wide uncertainty.",
    "Only 37 of 1000 applicants in the whole dataset are not foreign workers; that group is very small.",
    "'Sex' is derived from a combined marital-status/sex field; the code 'female : single' never occurs in the data, so single women cannot be distinguished.",
    "Removing sensitive attributes from the model ('fairness through unawareness') does not remove proxies: features such as job, housing or age can correlate with them.",
    "Passing these thresholds does not prove the model is fair. Different fairness definitions can conflict and no single metric is sufficient.",
    "The data come from one German bank in the 1970s-90s and may not represent any current population.",
]


def _status(gap: float) -> str:
    if gap < ACCEPTABLE:
        return "ACCEPTABLE"
    if gap < ATTENTION:
        return "NEEDS ATTENTION"
    return "CONCERNING"


@lru_cache(maxsize=1)
def fairness_report() -> dict:
    bundle = get_model()
    test = bundle.test_set
    y_true = test[TARGET].to_numpy()
    proba = bundle.pipeline.predict_proba(test[MODEL_FEATURES])[:, 1]
    y_pred = (proba >= bundle.threshold).astype(int)

    attributes = []
    for attr, desc in ATTRIBUTES.items():
        sf = test[attr]
        mf = MetricFrame(
            metrics={
                "count": count, "high_risk_rate": selection_rate,
                "fpr": false_positive_rate, "fnr": false_negative_rate,
                "accuracy": accuracy_score,
            },
            y_true=y_true, y_pred=y_pred, sensitive_features=sf,
        )
        groups = []
        for g, row in mf.by_group.iterrows():
            mask = (sf == g).to_numpy()
            groups.append({
                "group": str(g), "count": int(row["count"]),
                "approval_rate": round(1 - float(row["high_risk_rate"]), 4),
                "actual_good_rate": round(1 - float(y_true[mask].mean()), 4),
                "false_positive_rate": round(float(row["fpr"]), 4),
                "false_negative_rate": round(float(row["fnr"]), 4),
                "accuracy": round(float(row["accuracy"]), 4),
            })
        dpd = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sf))
        dpr = float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sf))
        eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sf))
        gap_status = _status(max(dpd, eod))
        small = [g for g in groups if g["count"] < MIN_GROUP_SIZE]
        status = INSUFFICIENT if small else gap_status
        ci = fairness_gap_cis(y_true, y_pred, sf.to_numpy())

        warnings = [f"Group '{g['group']}' has only {g['count']} test applicants; its metrics are unreliable."
                    for g in small]
        hi = max(groups, key=lambda g: g["approval_rate"])
        lo = min(groups, key=lambda g: g["approval_rate"])
        interpretation = (
            f"Approval rate is {hi['approval_rate']:.0%} for '{hi['group']}' vs {lo['approval_rate']:.0%} "
            f"for '{lo['group']}' (gap {dpd:.0%}). The largest gap in error rates between groups is {eod:.0%}. "
        )
        if status == INSUFFICIENT:
            interpretation += (f"No verdict is given: at least one group has fewer than {MIN_GROUP_SIZE} test "
                               f"applicants, so the gaps could be chance (on the gaps alone the status would be "
                               f"{gap_status}).")
        elif status == "ACCEPTABLE":
            interpretation += "No large disparity was detected on these metrics, but this does not prove the model is fair."
        elif status == "NEEDS ATTENTION":
            interpretation += "There is a noticeable disparity that should be investigated before any real use."
        else:
            interpretation += "There is a large disparity; this model should not be used for this group without mitigation."
        if hi["actual_good_rate"] != lo["actual_good_rate"]:
            interpretation += (f" Note: the actual good-credit rate in the data also differs "
                               f"({hi['actual_good_rate']:.0%} vs {lo['actual_good_rate']:.0%}), "
                               "so part of the gap reflects the historical data.")
        attributes.append({
            "attribute": attr, "description": desc, "groups": groups,
            "demographic_parity_difference": round(dpd, 4),
            "demographic_parity_ratio": round(dpr, 4) if np.isfinite(dpr) else 0.0,
            "equalized_odds_difference": round(eod, 4),
            "demographic_parity_difference_ci": ci["demographic_parity_difference"],
            "equalized_odds_difference_ci": ci["equalized_odds_difference"],
            "status": status, "gap_status": gap_status,
            "interpretation": interpretation, "warnings": warnings,
        })

    order = ["ACCEPTABLE", "NEEDS ATTENTION", "CONCERNING"]
    reliable = [a["status"] for a in attributes if a["status"] != INSUFFICIENT]
    worst = max(reliable, key=order.index) if reliable else INSUFFICIENT
    return {
        "model_name": bundle.name,
        "evaluated_on": f"held-out test set ({len(test)} applicants), decision threshold {bundle.threshold}",
        "thresholds_used": {
            "acceptable_below": ACCEPTABLE, "needs_attention_below": ATTENTION,
            "rule": "status uses the larger of demographic parity difference and equalized odds difference",
            "min_group_size": MIN_GROUP_SIZE,
            "small_group_rule": (f"if any group has fewer than {MIN_GROUP_SIZE} test applicants the attribute is "
                                 f"marked {INSUFFICIENT} and excluded from the overall status"),
            "ci_method": "95% percentile bootstrap over the test set (1000 resamples, seed 42)",
        },
        "attributes": attributes,
        "overall_status": worst,
        "limitations": LIMITATIONS,
    }
