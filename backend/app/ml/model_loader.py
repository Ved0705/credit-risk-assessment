"""Loads trained artifacts once and keeps them in memory."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import settings
from .explainer import ShapExplainer


class ModelNotFoundError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: dict
    schema: dict
    explainer: ShapExplainer
    test_set: pd.DataFrame

    @property
    def threshold(self) -> float:
        return float(self.metadata["decision_threshold"])

    @property
    def name(self) -> str:
        return self.metadata["selected_model"]


@lru_cache(maxsize=1)
def get_model() -> ModelBundle:
    d = settings.model_dir
    required = ["model.joblib", "metadata.json", "feature_schema.json", "shap_background.csv", "test_set.csv"]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        raise ModelNotFoundError(
            f"Model artifacts missing in {d}: {missing}. Train first with: python -m app.ml.train"
        )
    try:
        pipeline = joblib.load(d / "model.joblib")
    except Exception as exc:  # e.g. scikit-learn version mismatch
        raise ModelNotFoundError(
            f"Could not load model ({exc.__class__.__name__}). Re-train with: python -m app.ml.train"
        ) from exc
    metadata = json.loads((d / "metadata.json").read_text())
    schema = json.loads((d / "feature_schema.json").read_text())
    background = pd.read_csv(d / "shap_background.csv")
    test_set = pd.read_csv(d / "test_set.csv")
    return ModelBundle(pipeline, metadata, schema, ShapExplainer(pipeline, background), test_set)
