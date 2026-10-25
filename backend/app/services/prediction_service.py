"""Risk prediction for a single applicant."""
from __future__ import annotations

import hashlib
import json
import uuid

import pandas as pd

from ..ml.model_loader import get_model
from ..ml.preprocessing import MODEL_FEATURES
from .explanation_service import explain_dataframe


def predict_probability(applicants: pd.DataFrame) -> list[float]:
    """Probability of default (class 1 = HIGH RISK) for each row."""
    bundle = get_model()
    return bundle.pipeline.predict_proba(applicants[MODEL_FEATURES])[:, 1].tolist()


def classify(p: float, threshold: float) -> tuple[str, str]:
    """Binary decision uses the cost-optimal threshold.

    Risk bands:
        LOW    p <  threshold                -> LOW_RISK
        MEDIUM threshold <= p < max(0.5, t)  -> HIGH_RISK but borderline (manual review)
        HIGH   p >= max(0.5, threshold)      -> HIGH_RISK
    """
    if p < threshold:
        return "LOW_RISK", "LOW"
    if p < max(0.5, threshold):
        return "HIGH_RISK", "MEDIUM"
    return "HIGH_RISK", "HIGH"


def input_fingerprint(applicant: dict) -> str:
    """Anonymous identifier: hash of the inputs (raw values are not stored)."""
    return hashlib.sha256(json.dumps(applicant, sort_keys=True).encode()).hexdigest()[:16]


def predict(applicant: dict, top_k: int = 5) -> dict:
    bundle = get_model()
    df = pd.DataFrame([applicant])
    p = predict_probability(df)[0]
    prediction, category = classify(p, bundle.threshold)
    factors = explain_dataframe(df)[:top_k]
    confidence = (1 - p) if prediction == "LOW_RISK" else p

    drivers = [f["label"] for f in factors if (f["direction"] == "increases_risk") == (prediction == "HIGH_RISK")][:2]
    summary = (
        f"The model estimates a {p:.0%} probability of default "
        f"(decision threshold {bundle.threshold:.0%}), so the applicant is classed as "
        f"{'HIGH' if prediction == 'HIGH_RISK' else 'LOW'} RISK"
        + (f", mainly because of: {', '.join(drivers)}." if drivers else ".")
    )
    return {
        "assessment_id": str(uuid.uuid4()),
        "prediction": prediction,
        "risk_category": category,
        "probability_of_default": round(p, 4),
        "confidence": round(confidence, 4),
        "risk_score": int(round(p * 100)),
        "decision_threshold": bundle.threshold,
        "model_name": bundle.name,
        "summary": summary,
        "top_factors": factors,
    }
