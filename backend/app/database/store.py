"""
In-memory store for assessment records (no database needed).

Records live only while the backend is running and reset on restart.
Stored per assessment (no raw applicant data, by design):
    assessment_id, timestamp, input_fingerprint (hash), model_name, prediction,
    risk_category, probability_of_default, top_factors (feature + direction + shap value)
"""
from __future__ import annotations

from datetime import datetime, timezone


class AssessmentStore:
    backend = "memory"

    def __init__(self) -> None:
        self.records: list[dict] = []

    def save(self, result: dict, fingerprint: str) -> bool:
        self.records.append({
            "assessment_id": result["assessment_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": fingerprint,
            "model_name": result["model_name"],
            "prediction": result["prediction"],
            "risk_category": result["risk_category"],
            "probability_of_default": result["probability_of_default"],
            "top_factors": [
                {"feature": f["feature"], "direction": f["direction"], "shap_value": f["shap_value"]}
                for f in result["top_factors"]
            ],
        })
        return True

    def recent(self, limit: int = 20) -> list[dict]:
        return [dict(r) for r in reversed(self.records[-limit:])]

    def clear(self) -> None:
        self.records.clear()

    def stats(self) -> dict:
        total = len(self.records)
        high = sum(r["prediction"] == "HIGH_RISK" for r in self.records)
        return {
            "total_assessments": total,
            "high_risk": high,
            "low_risk": total - high,
            "high_risk_pct": round(high / total, 4) if total else 0.0,
            "low_risk_pct": round((total - high) / total, 4) if total else 0.0,
            "by_category": {c: sum(r["risk_category"] == c for r in self.records) for c in ("LOW", "MEDIUM", "HIGH")},
            "storage": self.backend,
        }


_store = AssessmentStore()


def get_repository() -> AssessmentStore:
    return _store
