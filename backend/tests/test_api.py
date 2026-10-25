"""API tests. Run from backend/:  pytest -q   (requires a trained model in backend/models)."""
import copy

import pytest
from fastapi.testclient import TestClient

from app.database.store import get_repository
from app.main import app

EXAMPLE = {
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
RISKY = {**EXAMPLE, "checking_status": "< 0 DM", "duration": 60, "credit_amount": 15000,
         "credit_history": "no credits / all paid back duly", "savings_status": "< 100 DM",
         "employment": "< 1 year", "age": 22, "housing": "rent", "property": "unknown / no property"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_low_and_high(client):
    low = client.post("/predict", json=EXAMPLE).json()
    high = client.post("/predict", json=RISKY).json()
    assert low["prediction"] == "LOW_RISK"
    assert high["prediction"] == "HIGH_RISK"
    assert 0 <= low["probability_of_default"] <= 1
    assert high["probability_of_default"] > low["probability_of_default"]
    assert len(low["top_factors"]) == 5


def test_missing_field(client):
    bad = copy.deepcopy(EXAMPLE)
    del bad["age"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert any(e["field"] == "age" for e in r.json()["detail"])


def test_invalid_category(client):
    r = client.post("/predict", json={**EXAMPLE, "housing": "castle"})
    assert r.status_code == 422


def test_out_of_range(client):
    assert client.post("/predict", json={**EXAMPLE, "credit_amount": -5}).status_code == 422


def test_unknown_field_rejected(client):
    """Sensitive fields that the model does not use must not be accepted."""
    assert client.post("/predict", json={**EXAMPLE, "foreign_worker": "yes"}).status_code == 422


def test_explain_has_all_features(client):
    r = client.post("/explain", json=EXAMPLE).json()
    assert len(r["contributions"]) == 18
    for c in r["contributions"]:
        assert c["direction"] in ("increases_risk", "decreases_risk")
        assert (c["shap_value"] > 0) == (c["direction"] == "increases_risk")


def test_metrics_and_features(client):
    m = client.get("/model/metrics").json()
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(m["selected_metrics"])
    assert len(m["comparison"]) == 3
    f = client.get("/model/features").json()
    assert set(f["numeric"]) | set(f["categorical"]) == set(EXAMPLE)


def test_fairness(client):
    r = client.get("/fairness").json()
    attrs = {a["attribute"] for a in r["attributes"]}
    assert {"sex", "age_group", "foreign_worker_group"} == attrs
    assert r["limitations"]


def test_stability(client):
    r = client.post("/stability", json={"applicant": EXAMPLE, "n_samples": 10}).json()
    assert r["n_samples"] == 10
    assert 0 <= r["same_prediction"] <= 10
    assert r["stability_score"] == r["same_prediction"] / 10


def test_store_does_not_keep_raw_inputs(client):
    get_repository().clear()
    client.post("/predict", json=EXAMPLE)
    doc = client.get("/assessments").json()[0]
    assert "age" not in doc and "credit_amount" not in doc  # raw inputs are not stored
    assert client.get("/assessments/stats").json()["total_assessments"] == 1


def test_recent_twice_in_memory(client):
    """Regression: reading history must not mutate stored timestamps."""
    client.post("/predict", json=EXAMPLE)
    assert client.get("/assessments").status_code == 200
    assert client.get("/assessments").status_code == 200


def test_cors_allows_other_local_ports(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5174"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5174"


def test_comparison_at_cost_threshold(client):
    """Every model is also evaluated at its own cost-optimal threshold (chosen on training data)."""
    m = client.get("/model/metrics").json()
    for row in m["comparison"]:
        assert 0.05 <= row["cost_threshold"] <= 0.95
        assert row["at_cost_threshold"]["threshold"] == row["cost_threshold"]
    sel = next(r for r in m["comparison"] if r["model"] == m["selected_model"])
    assert sel["cost_threshold"] == m["decision_threshold"]
    assert sel["at_cost_threshold"] == m["selected_metrics"]


def test_metric_confidence_intervals(client):
    m = client.get("/model/metrics").json()
    for k, (lo, hi) in m["selected_metrics_ci"].items():
        assert lo <= m["selected_metrics"][k] <= hi, k


def test_fairness_small_groups_get_no_verdict(client):
    r = client.get("/fairness").json()
    by = {a["attribute"]: a for a in r["attributes"]}
    fw = by["foreign_worker_group"]  # only 6 non-foreign workers in the test set
    assert fw["status"] == "INSUFFICIENT DATA" and fw["warnings"]
    assert r["overall_status"] != "INSUFFICIENT DATA"
    for a in r["attributes"]:
        lo, hi = a["equalized_odds_difference_ci"]
        assert lo <= hi
        if min(g["count"] for g in a["groups"]) >= 30:
            assert a["status"] == a["gap_status"]
