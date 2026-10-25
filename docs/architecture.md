# Architecture

## Components

| Component | Responsibility |
|---|---|
| **React frontend** (`frontend/`) | Dashboard UI. Loads the input schema from the API, sends applicant data, renders verdict, SHAP charts, fairness and stability results. |
| **FastAPI backend** (`backend/app/main.py`) | REST endpoints, Pydantic validation, CORS, uniform JSON errors. |
| **Services** (`backend/app/services/`) | Business logic: prediction, SHAP explanation, fairness audit, stability analysis. |
| **ML package** (`backend/app/ml/`) | Data decoding, preprocessing pipeline, training, evaluation, SHAP wrapper, model loading. |
| **Store** (`backend/app/database/store.py`) | Keeps assessment records in memory while the backend runs. |

## Request flow — `POST /predict`

```text
Browser form ─▶ api.predict()  ─▶  FastAPI route /predict
   │                                   │ 1. Pydantic validates types, ranges, allowed categories (422 otherwise)
   │                                   │ 2. prediction_service.predict()
   │                                   │      ├─ Pipeline.predict_proba  (OneHot + Scale → Random Forest)
   │                                   │      ├─ classify(p, threshold) → LOW_RISK/HIGH_RISK + band
   │                                   │      └─ explanation_service → SHAP top factors
   │                                   │ 3. repository.save()  (no raw inputs, only hash + result)
   ◀───────────── JSON response ───────┘
```

## Startup

`model_loader.get_model()` loads `model.joblib`, `metadata.json`, `feature_schema.json`, the SHAP background sample and the test set once (cached). If anything is missing, prediction endpoints return `503 model_not_available` with the command to train; `/health` reports `degraded`.

## Why this design

- **Monolith with service modules**: easy to run and explain; no microservices needed for a single model.
- **Schema-driven form**: `GET /model/features` is generated from the same code book used in training, so UI fields cannot diverge from model features.
- **Artifacts on disk**: training is a separate script (`python -m app.ml.train`); the API only loads results. The fairness audit and global importance are computed from saved artifacts, so they are reproducible.
- **No database**: history is kept in memory, so there is nothing extra to install or run.

## Data stored per assessment

`assessment_id, timestamp, input_fingerprint (SHA-256 prefix of inputs), model_name, prediction, risk_category, probability_of_default, top_factors[feature, direction, shap_value]` — no age, amounts or other raw inputs.
