"""Request schemas. Fields mirror EXACTLY the features the model was trained on."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..ml.preprocessing import CODEBOOK

# Literal types built from the official code book -> invalid categories are rejected (422)
CheckingStatus = Literal[tuple(CODEBOOK["checking_status"].values())]  # type: ignore[valid-type]
CreditHistory = Literal[tuple(CODEBOOK["credit_history"].values())]  # type: ignore[valid-type]
Purpose = Literal[tuple(CODEBOOK["purpose"].values())]  # type: ignore[valid-type]
Savings = Literal[tuple(CODEBOOK["savings_status"].values())]  # type: ignore[valid-type]
Employment = Literal[tuple(CODEBOOK["employment"].values())]  # type: ignore[valid-type]
OtherDebtors = Literal[tuple(CODEBOOK["other_debtors"].values())]  # type: ignore[valid-type]
Property = Literal[tuple(CODEBOOK["property"].values())]  # type: ignore[valid-type]
OtherPlans = Literal[tuple(CODEBOOK["other_installment_plans"].values())]  # type: ignore[valid-type]
Housing = Literal[tuple(CODEBOOK["housing"].values())]  # type: ignore[valid-type]
Job = Literal[tuple(CODEBOOK["job"].values())]  # type: ignore[valid-type]
Telephone = Literal[tuple(CODEBOOK["telephone"].values())]  # type: ignore[valid-type]


class ApplicantInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "checking_status": "no checking account", "duration": 24,
                "credit_history": "existing credits paid back duly till now",
                "purpose": "car (used)", "credit_amount": 3500,
                "savings_status": "100 - 500 DM", "employment": "1 - 4 years",
                "installment_rate": 3, "other_debtors": "none", "residence_since": 2,
                "property": "car or other property", "age": 35,
                "other_installment_plans": "none", "housing": "own",
                "existing_credits": 1, "job": "skilled employee / official",
                "num_dependents": 1, "telephone": "none",
            }
        },
    )

    checking_status: CheckingStatus
    duration: int = Field(..., ge=4, le=72, description="Loan duration in months")
    credit_history: CreditHistory
    purpose: Purpose
    credit_amount: float = Field(..., ge=250, le=20000, description="Credit amount in Deutsche Mark")
    savings_status: Savings
    employment: Employment
    installment_rate: int = Field(..., ge=1, le=4, description="Installment as % of disposable income (1-4)")
    other_debtors: OtherDebtors
    residence_since: int = Field(..., ge=1, le=4)
    property: Property
    age: int = Field(..., ge=18, le=75)
    other_installment_plans: OtherPlans
    housing: Housing
    existing_credits: int = Field(..., ge=1, le=4)
    job: Job
    num_dependents: int = Field(..., ge=1, le=2)
    telephone: Telephone


class StabilityRequest(BaseModel):
    applicant: ApplicantInput
    n_samples: int = Field(20, ge=5, le=200, description="Number of perturbed copies")
    noise_level: float = Field(0.05, gt=0, le=0.3, description="Relative perturbation size (0.05 = ±5%)")
    seed: int = Field(42, description="Random seed for reproducibility")
