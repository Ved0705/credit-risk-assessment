from fastapi import APIRouter

from ..ml.model_loader import get_model

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/metrics", summary="Evaluation metrics and model comparison")
def metrics():
    m = get_model().metadata
    return {k: m[k] for k in (
        "selected_model", "selection_reason", "decision_threshold", "threshold_reason",
        "selected_metrics", "comparison", "dataset", "trained_at",
    )} | {k: m.get(k) for k in ("selected_metrics_ci", "ci_method")}


@router.get("/features", summary="Input schema: feature names, allowed values and ranges")
def features():
    b = get_model()
    return {**b.schema, "excluded_sensitive": b.metadata["features"]["excluded_sensitive"]}


@router.get("/importance", summary="Global SHAP feature importance (mean |SHAP| on the test set)")
def importance():
    m = get_model().metadata
    labels = get_model().schema["labels"]
    return {
        "output_units": m["shap_output_units"],
        "importance": [{**g, "label": labels[g["feature"]]} for g in m["global_importance"]],
    }
