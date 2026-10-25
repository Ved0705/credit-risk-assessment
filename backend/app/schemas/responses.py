"""Response schemas (shown in Swagger at /docs)."""
from typing import Literal, Optional

from pydantic import BaseModel


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: str | float | int
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]
    explanation: str


class PredictionResponse(BaseModel):
    assessment_id: str
    prediction: Literal["LOW_RISK", "HIGH_RISK"]
    risk_category: Literal["LOW", "MEDIUM", "HIGH"]
    probability_of_default: float
    confidence: float
    risk_score: int
    decision_threshold: float
    model_name: str
    summary: str
    top_factors: list[FeatureContribution]
    saved: bool


class ExplanationResponse(BaseModel):
    prediction: Literal["LOW_RISK", "HIGH_RISK"]
    probability_of_default: float
    base_value: float
    output_units: str
    contributions: list[FeatureContribution]
    how_to_read: str


class GroupMetrics(BaseModel):
    group: str
    count: int
    approval_rate: float
    actual_good_rate: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float


class AttributeFairness(BaseModel):
    attribute: str
    description: str
    groups: list[GroupMetrics]
    demographic_parity_difference: float
    demographic_parity_ratio: float
    equalized_odds_difference: float
    demographic_parity_difference_ci: Optional[list[float]] = None
    equalized_odds_difference_ci: Optional[list[float]] = None
    status: Literal["ACCEPTABLE", "NEEDS ATTENTION", "CONCERNING", "INSUFFICIENT DATA"]
    gap_status: Literal["ACCEPTABLE", "NEEDS ATTENTION", "CONCERNING"]
    interpretation: str
    warnings: list[str]


class FairnessResponse(BaseModel):
    model_name: str
    evaluated_on: str
    thresholds_used: dict
    attributes: list[AttributeFairness]
    overall_status: str
    limitations: list[str]


class FeatureSensitivity(BaseModel):
    feature: str
    label: str
    original_value: float
    tested_values: list[float]
    probabilities: list[float]
    max_probability_change: float
    flips_decision: bool


class StabilityResponse(BaseModel):
    original_prediction: Literal["LOW_RISK", "HIGH_RISK"]
    original_probability: float
    n_samples: int
    noise_level: float
    same_prediction: int
    stability_score: float
    status: Literal["STABLE", "MODERATELY STABLE", "UNSTABLE"]
    probability_mean: float
    probability_std: float
    probability_min: float
    probability_max: float
    sensitive_features: list[FeatureSensitivity]
    perturbed_features: list[str]
    interpretation: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    database: str
    detail: Optional[str] = None
