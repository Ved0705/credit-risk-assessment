from fastapi import APIRouter, Query

from ..database.store import get_repository
from ..schemas.prediction import ApplicantInput, StabilityRequest
from ..schemas.responses import ExplanationResponse, PredictionResponse, StabilityResponse
from ..services import explanation_service, prediction_service, stability_service

router = APIRouter(tags=["Assessment"])


@router.post("/predict", response_model=PredictionResponse, summary="Predict credit risk for one applicant")
def predict(applicant: ApplicantInput):
    data = applicant.model_dump()
    result = prediction_service.predict(data)
    result["saved"] = get_repository().save(result, prediction_service.input_fingerprint(data))
    return result


@router.post("/explain", response_model=ExplanationResponse, summary="Full SHAP explanation for one applicant")
def explain(applicant: ApplicantInput):
    data = applicant.model_dump()
    res = prediction_service.predict(data, top_k=0)
    return explanation_service.explain(data, res["probability_of_default"], res["prediction"])


@router.post("/stability", response_model=StabilityResponse, summary="Prediction stability under small input changes")
def stability(req: StabilityRequest):
    return stability_service.analyse(req.applicant.model_dump(), req.n_samples, req.noise_level, req.seed)


@router.get("/assessments", summary="Most recent stored assessments")
def assessments(limit: int = Query(20, ge=1, le=100)):
    return get_repository().recent(limit)


@router.get("/assessments/stats", summary="Aggregate statistics for the dashboard")
def assessment_stats():
    return get_repository().stats()
