from fastapi import APIRouter

from ..schemas.responses import FairnessResponse
from ..services.fairness_service import fairness_report

router = APIRouter(tags=["Fairness"])


@router.get("/fairness", response_model=FairnessResponse, summary="Group fairness audit on the test set")
def fairness():
    return fairness_report()
