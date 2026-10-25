"""
SHAP wrapper shared by training (global importance) and the API (local explanations).

The model sees one-hot encoded columns (e.g. `purpose_car (new)`), but users think in
terms of original features (e.g. `purpose`). SHAP values are additive, so we can simply
SUM the SHAP values of all one-hot columns that belong to the same original feature.

Sign convention (target = 1 means HIGH RISK / bad credit):
    shap > 0  -> pushes the prediction TOWARDS high risk
    shap < 0  -> pushes the prediction TOWARDS low risk
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


class ShapExplainer:
    def __init__(self, pipeline: Pipeline, background: pd.DataFrame):
        self.pipeline = pipeline
        self.pre = pipeline.named_steps["preprocessor"]
        self.clf = pipeline.named_steps["classifier"]
        bg = self.pre.transform(background[MODEL_FEATURES])
        self.columns = list(self.pre.get_feature_names_out())
        self.groups = self._group_index()

        name = type(self.clf).__name__
        if name == "LogisticRegression":
            self.explainer = shap.LinearExplainer(self.clf, bg)
            self.output_units = "log-odds"
        else:  # RandomForest / XGBoost
            self.explainer = shap.TreeExplainer(self.clf, data=bg, model_output="raw" if name.startswith("XGB") else "probability")
            self.output_units = "log-odds" if name.startswith("XGB") else "probability"

        ev = np.atleast_1d(self.explainer.expected_value)
        self.base_value = float(ev[-1])  # last entry = class 1 when two are returned

    def _group_index(self) -> dict[str, list[int]]:
        """Map each original feature to the indices of its transformed columns."""
        groups: dict[str, list[int]] = {f: [] for f in MODEL_FEATURES}
        ohe = self.pre.named_transformers_["cat"]
        expected = list(NUMERIC_FEATURES)
        for feat, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
            expected += [(feat, c) for c in cats]
        for i, item in enumerate(expected):
            groups[item if isinstance(item, str) else item[0]].append(i)
        assert len(expected) == len(self.columns), "feature mapping mismatch"
        return groups

    def raw_shap(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X[MODEL_FEATURES])
        vals = self.explainer.shap_values(Xt)
        if isinstance(vals, list):  # older SHAP: list per class
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[:, :, 1]
        return vals

    def grouped_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values aggregated per ORIGINAL feature, shape (n_samples, n_features)."""
        vals = self.raw_shap(X)
        return pd.DataFrame(
            {f: vals[:, idx].sum(axis=1) for f, idx in self.groups.items()},
            index=X.index,
        )

    def global_importance(self, X: pd.DataFrame) -> list[dict]:
        g = self.grouped_shap(X).abs().mean().sort_values(ascending=False)
        return [{"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in g.items()]
