"""Local SHAP explanations for a single applicant."""
from __future__ import annotations

import pandas as pd

from ..ml.model_loader import get_model
from ..ml.preprocessing import FEATURE_LABELS, MODEL_FEATURES

HOW_TO_READ = (
    "Each factor shows how much it moved this applicant's predicted risk away from the "
    "average applicant. A POSITIVE value pushes the decision towards HIGH RISK; a NEGATIVE "
    "value pushes it towards LOW RISK. Bigger bars = stronger influence. These describe what "
    "the model learned from historical data, not causal facts about the applicant."
)


def _format_value(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v.item() if hasattr(v, "item") else v


def explain_dataframe(df: pd.DataFrame) -> list[dict]:
    """Return per-feature SHAP contributions for the first row, sorted by impact."""
    bundle = get_model()
    shap_row = bundle.explainer.grouped_shap(df[MODEL_FEATURES]).iloc[0]
    row = df.iloc[0]
    out = []
    for feat, val in shap_row.items():
        label = FEATURE_LABELS[feat]
        value = _format_value(row[feat])
        up = val > 0
        out.append({
            "feature": feat,
            "label": label,
            "value": value,
            "shap_value": round(float(val), 4),
            "direction": "increases_risk" if up else "decreases_risk",
            "explanation": f"{label} = '{value}' {'increases' if up else 'reduces'} the estimated risk of default.",
        })
    out.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return out


def explain(applicant: dict, probability: float, prediction: str) -> dict:
    bundle = get_model()
    contributions = explain_dataframe(pd.DataFrame([applicant]))
    return {
        "prediction": prediction,
        "probability_of_default": round(probability, 4),
        "base_value": round(bundle.explainer.base_value, 4),
        "output_units": bundle.explainer.output_units,
        "contributions": contributions,
        "how_to_read": HOW_TO_READ,
    }
