"""ML pipeline tests: preprocessing, prediction output, explanation additivity."""
import numpy as np
import pandas as pd

from app.config import settings
from app.ml.model_loader import get_model
from app.ml.preprocessing import MODEL_FEATURES, SENSITIVE_COLUMNS, TARGET, build_preprocessor, load_dataset
from app.services.prediction_service import classify


def test_dataset_decoding():
    df = load_dataset(settings.data_path)
    assert len(df) == 1000
    assert df[TARGET].isin([0, 1]).all()
    assert abs(df[TARGET].mean() - 0.30) < 1e-9  # 300 bad / 700 good in the original data
    assert not df[MODEL_FEATURES].isna().any().any()
    assert not df["checking_status"].str.startswith("A").any()  # codes were decoded


def test_sensitive_columns_not_model_features():
    assert not set(SENSITIVE_COLUMNS) & set(MODEL_FEATURES)


def test_preprocessor_shapes():
    df = load_dataset(settings.data_path)
    Xt = build_preprocessor().fit_transform(df[MODEL_FEATURES])
    assert Xt.shape[0] == 1000 and not np.isnan(Xt).any()


def test_model_loads_and_predicts_probabilities():
    b = get_model()
    p = b.pipeline.predict_proba(b.test_set[MODEL_FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_shap_additivity():
    """base value + sum of SHAP values must reproduce the model output."""
    b = get_model()
    X = b.test_set[MODEL_FEATURES].head(5)
    shap_sum = b.explainer.grouped_shap(X).sum(axis=1).to_numpy() + b.explainer.base_value
    p = b.pipeline.predict_proba(X)[:, 1]
    target = p if b.explainer.output_units == "probability" else np.log(p / (1 - p))
    # interventional TreeSHAP with a 100-row background is accurate to < 0.01 here
    assert np.allclose(shap_sum, target, atol=0.01)


def test_classify_bands():
    assert classify(0.10, 0.27) == ("LOW_RISK", "LOW")
    assert classify(0.35, 0.27) == ("HIGH_RISK", "MEDIUM")
    assert classify(0.80, 0.27) == ("HIGH_RISK", "HIGH")


def test_bootstrap_ci_contains_estimate():
    from app.ml.uncertainty import fairness_gap_cis, metric_cis
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    ci = metric_cis(y, p, 0.5, n_boot=200)
    from sklearn.metrics import roc_auc_score
    lo, hi = ci["roc_auc"]
    assert lo <= roc_auc_score(y, p) <= hi
    g = np.where(rng.random(300) < 0.5, "a", "b")
    f = fairness_gap_cis(y, p >= 0.5, g, n_boot=200)
    assert f["valid_resamples"] == 200 and f["equalized_odds_difference"][0] <= f["equalized_odds_difference"][1]
