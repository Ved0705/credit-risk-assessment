# CreditLens — Explainable Credit Risk Assessment System

An end-to-end machine-learning web application that predicts the credit risk of a loan applicant **and explains the decision**: why the model decided (SHAP), whether it treats groups differently (Fairlearn), and whether the decision holds up under small input changes (stability test).

> ⚠️ **Academic project.** It is trained on a small historical dataset and must not be used for real lending decisions.

---

## 1. Project overview

Most credit-scoring demos return only "Approved" or "Rejected". CreditLens returns:

| Output | Example |
|---|---|
| Prediction | `LOW_RISK` / `HIGH_RISK` |
| Probability of default | 41.2 % |
| Risk band | LOW / MEDIUM (borderline, needs review) / HIGH |
| Local explanation | "Checking account balance `0–200 DM` +0.050 → increases risk" |
| Global importance | Mean \|SHAP\| for every feature |
| Fairness audit | Approval-rate and error-rate gaps by sex, age group and foreign-worker status |
| Stability | "20/20 perturbed copies got the same decision (100 %)" |

## 2. Problem statement

Automated credit decisions affect people's access to money. A black-box model that cannot justify its decisions is hard to trust, hard to audit for discrimination and hard to contest. This project demonstrates how an ML credit model can be made **transparent, auditable and honest about its limits**.

## 3. Objectives

1. Train and compare several classifiers and choose one with a defensible, documented procedure.
2. Evaluate with risk-appropriate metrics, not accuracy alone.
3. Explain individual and global predictions with SHAP.
4. Audit group fairness with Fairlearn and report limitations openly.
5. Test prediction stability under small input perturbations.
6. Package everything as a usable web application with a REST API.

## 4. Features

- **Dashboard** — total assessments, low/high-risk share, model accuracy, ROC-AUC, recall, fairness summary, recent assessments.
- **Credit Assessment** — plain-language key reasons (factors raising / lowering risk), printable assessment report, field-level validation, schema-driven applicant form (fields are loaded from the model, so UI and model never drift apart), example applicants, verdict with probability meter and threshold, top SHAP factors, one-click stability test.
- **Explainability** — local SHAP chart + table for the current applicant, global SHAP importance.
- **Fairness** — per-group approval rate, FPR, FNR, accuracy, demographic parity and equalized odds with 95 % bootstrap intervals, status (INSUFFICIENT DATA when a group has < 30 test applicants) and plain-language interpretation, explicit limitations.
- **Model Performance** — metrics with 95 % bootstrap confidence intervals, confusion matrix, model comparison at 0.5 **and at each model's own cost-optimal threshold**, "why accuracy is not enough".
- **REST API** with Swagger docs, Pydantic validation and friendly errors.
- **Assessment history** kept in memory while the backend runs (no database needed; resets on restart). Raw applicant data is never stored — only the decision, the top factors and an anonymous hash of the input.

## 5. Architecture

```text
 React + Vite dashboard  ──HTTP/JSON──▶  FastAPI (Pydantic validation)
                                              │
                    ┌─────────────────────────┼───────────────────────────┐
                    ▼                         ▼                           ▼
          prediction_service        explanation_service           fairness_service
          (sklearn Pipeline:        (SHAP TreeExplainer,          (Fairlearn MetricFrame
           preprocessing +           one-hot values summed         on held-out test set)
           Random Forest)            per original feature)
                    │                                              stability_service
                    ▼                                              (perturb + re-predict)
              in-memory assessment history
```

More detail: [`docs/architecture.md`](docs/architecture.md).

## 6. Tech stack

| Layer | Technology |
|---|---|
| ML | Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Fairlearn |
| Backend | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React 19, Vite, React Router, Recharts |
| Testing | pytest, Vitest, Testing Library |
| DevOps | Docker, Docker Compose |

## 7. Dataset

**German Credit Data (Statlog)** — Prof. Hans Hofmann, UCI Machine Learning Repository. 1,000 applicants, 20 attributes, target good (700) / bad (300) credit. The original coded file (`A11`, `A43`, …) is in `data/german_credit_raw.csv` and is decoded in code using the official UCI code book. Full feature description: [`data/README.md`](data/README.md).

| Decision | Reason |
|---|---|
| Target `is_bad` = 1 for bad credit | We predict the risk of default (HIGH RISK is the positive class). |
| `personal_status_sex` **excluded** from the model | Contains sex (a protected attribute) mixed with marital status. Kept only for the fairness audit. |
| `foreign_worker` **excluded** from the model | National origin is a protected attribute. Kept only for the fairness audit. |
| `age` **kept** as a feature, but audited | Commonly used in credit scoring; we measure its fairness impact explicitly. |
| All other 17 features kept | No column is dropped without a reason. |
| No imputation needed | The dataset has no missing values; the pipeline still handles them (median / mode). |

## 8. ML methodology

```text
Raw coded CSV → decode with UCI code book → clean (dedupe, types, missing values)
→ stratified 80/20 train/test split (seed 42)
→ Pipeline[ ColumnTransformer(StandardScaler | OneHotEncoder) → classifier ]
→ GridSearchCV, 5-fold stratified, scoring = ROC-AUC (training data only)
→ select model → cost-based threshold → evaluate on test set → save artifacts
```

- Preprocessing is **inside** the pipeline, so it is fitted on training folds only — no data leakage.
- **Model selection** uses cross-validated ROC-AUC on the training set. If a simpler model is within 0.01 AUC of the best one, the simpler model is preferred (order: Logistic Regression < Random Forest < XGBoost).
- **Decision threshold**: the dataset documentation defines a cost matrix — approving a bad applicant costs 5, rejecting a good one costs 1. We choose the threshold minimising this cost on *out-of-fold training predictions* (not on the test set).
- **Fair comparison**: every candidate model gets its own cost-optimal threshold the same way, and is reported on the test set both at 0.5 and at that threshold (`comparison[].at_cost_threshold` in `metadata.json`). Selection itself still uses CV ROC-AUC only.
- **Uncertainty**: test metrics carry 95 % bootstrap confidence intervals (`selected_metrics_ci`); with 200 test applicants, ROC-AUC 0.809 has a range of about 0.74–0.87, so the three models are not statistically distinguishable.

Details: [`docs/methodology.md`](docs/methodology.md).

## 9. Explainability methodology

- `shap.TreeExplainer` (interventional, 100-row training background) for tree models; `LinearExplainer` if Logistic Regression is selected.
- One-hot SHAP values are **summed back** to the original feature (SHAP values are additive), so users see "Purpose" rather than "purpose_car (new)".
- Sign convention: **positive → pushes towards HIGH RISK, negative → towards LOW RISK**. For the Random Forest, values are in probability units: base value (≈ 30 %) + sum of SHAP values ≈ predicted probability (verified in tests to within 0.01).
- Global importance = mean |SHAP| on the 200 test applicants.

## 10. Fairness methodology

- Groups: **sex** (derived), **age** (< 25 vs ≥ 25), **foreign-worker status**.
- Predictions on the held-out test set at the model's decision threshold.
- Fairlearn `MetricFrame` per group: approval rate, false positive rate (good payers wrongly rejected), false negative rate (defaulters missed), accuracy, plus the actual good-credit rate.
- Summary metrics: demographic parity difference and ratio, equalized odds difference.
- Status rule of thumb: max(DPD, EOD) < 0.10 → ACCEPTABLE, < 0.20 → NEEDS ATTENTION, otherwise CONCERNING. This is a heuristic, not a legal standard, and "acceptable" never means "proven fair".
- If any group has fewer than 30 test applicants the attribute is marked **INSUFFICIENT DATA** (no verdict) and left out of the overall status — e.g. only 6 test applicants are not foreign workers.
- Each gap has a 95 % percentile-bootstrap interval (1,000 resamples of the test set, seed 42).
- No reweighting or resampling was applied — the numbers show the model as it is.

## 11. Stability methodology

For one applicant:
1. Generate N (default 20) copies with loan **duration** and **credit amount** multiplied by `1 + Normal(0, 5 %)` (clipped to valid ranges); re-predict; stability = share with the same decision.
2. One feature at a time, nudge each non-sensitive numeric feature (±5 %/±10 % for continuous, ±1 step for small integer scales) and report the probability change and whether the decision flips.
3. Age is not perturbed: it is a sensitive attribute, so changing it is a fairness question rather than a robustness question.

Status: ≥ 90 % STABLE, ≥ 70 % MODERATELY STABLE, otherwise UNSTABLE.

## 12. API documentation

Interactive Swagger UI: **http://localhost:8000/docs** (ReDoc at `/redoc`).

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API, model and database status |
| POST | `/predict` | Prediction, probability, risk band, top 5 SHAP factors; stores the record |
| POST | `/explain` | SHAP contributions for all 18 features |
| POST | `/stability` | Stability test (`{"applicant": {...}, "n_samples": 20, "noise_level": 0.05}`) |
| GET | `/fairness` | Group fairness audit |
| GET | `/model/metrics` | Selected-model metrics, comparison table, dataset info |
| GET | `/model/features` | Input schema (ranges, allowed categories, labels) |
| GET | `/model/importance` | Global SHAP importance |
| GET | `/assessments` | Recent stored assessments |
| GET | `/assessments/stats` | Dashboard aggregates |

Example request (`POST /predict`):

```json
{
  "checking_status": "no checking account", "duration": 24,
  "credit_history": "existing credits paid back duly till now",
  "purpose": "car (used)", "credit_amount": 3500,
  "savings_status": "100 - 500 DM", "employment": "1 - 4 years",
  "installment_rate": 3, "other_debtors": "none", "residence_since": 2,
  "property": "car or other property", "age": 35,
  "other_installment_plans": "none", "housing": "own",
  "existing_credits": 1, "job": "skilled employee / official",
  "num_dependents": 1, "telephone": "none"
}
```

Example response (abridged, real output):

```json
{
  "prediction": "LOW_RISK", "risk_category": "LOW",
  "probability_of_default": 0.1045, "confidence": 0.8955, "risk_score": 10,
  "decision_threshold": 0.27, "model_name": "Random Forest",
  "top_factors": [
    {"feature": "checking_status", "label": "Checking account balance",
     "value": "no checking account", "shap_value": -0.128, "direction": "decreases_risk",
     "explanation": "Checking account balance = 'no checking account' reduces the estimated risk of default."}
  ]
}
```

Errors are JSON without stack traces: `422 invalid_input` (lists each bad field), `503 model_not_available`, `500 internal_error`.

## 13. Project structure

```text
credit-risk-assessment/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, error handlers
│   │   ├── config.py            # settings from env / .env
│   │   ├── routes/              # prediction.py, model.py, fairness.py
│   │   ├── schemas/             # Pydantic request & response models
│   │   ├── services/            # prediction, explanation, fairness, stability
│   │   ├── ml/                  # preprocessing, train, evaluate, explainer, model_loader
│   │   └── database/store.py    # in-memory assessment history
│   ├── models/                  # trained artifacts (model.joblib, metadata.json, …)
│   ├── tests/                   # pytest
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/{components,pages,services,test}/
│   ├── package.json
│   └── Dockerfile
├── data/                        # raw dataset + description
├── notebooks/model_analysis.ipynb
├── docs/{architecture.md, methodology.md}
├── docker-compose.yml
└── README.md
```

## 14. Installation

Prerequisites: **Python 3.11–3.13** (pinned library versions need ≥ 3.11), **Node.js 20.19+ or 22.12+**, optionally Docker. No database needed.

```bash
# Backend
cd backend
py -3.11 -m venv .venv      # needs Python 3.11-3.13; check with: py --list
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt   # notebook extras: requirements-notebook.txt
copy .env.example .env        # macOS/Linux: cp .env.example .env
python -m app.ml.train        # ~30 s; re-creates the model for your library versions

# Frontend
cd ../frontend
npm install
```

> Always run `python -m app.ml.train` once after installing — a saved scikit-learn model may not load under a different scikit-learn version.

## 15. Running locally

```bash
# Terminal 1 – backend
cd backend
.venv\Scripts\activate          # or: source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 – frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** (UI) and **http://localhost:8000/docs** (API).

Tests:

```bash
cd backend && pytest -q        # 23 tests
cd frontend && npm test        # 9 tests
```

## 16. Docker

```bash
docker compose up --build
```

Starts the backend (8000; the model is trained during the image build) and the frontend (5173).

## 17. Screenshots

_Add screenshots of your own run here:_

- `docs/screenshots/dashboard.png`
- `docs/screenshots/assessment.png`
- `docs/screenshots/explainability.png`
- `docs/screenshots/fairness.png`
- `docs/screenshots/performance.png`

## 18. Results

Held-out test set: 200 applicants (60 bad, 140 good). Seed 42. Numbers come from `backend/models/metadata.json` and may shift slightly with different library versions.

**Model comparison (test set, threshold 0.5)**

| Model | CV ROC-AUC (train) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7749 | 0.770 | 0.694 | 0.417 | 0.521 | 0.809 |
| Random Forest | 0.7923 | 0.765 | 0.724 | 0.350 | 0.472 | 0.809 |
| XGBoost | 0.7987 | 0.775 | 0.674 | 0.483 | 0.563 | 0.802 |

**Selected: Random Forest** (max_depth 8, min_samples_leaf 5, 300 trees). XGBoost had the best CV AUC, but Random Forest was within 0.01 and is the simpler model, so it was preferred. All three models perform similarly on this small dataset.

**Selected model at cost-optimal threshold 0.27**

| Accuracy | Precision | Recall | F1 | ROC-AUC | Cost (5·FN + FP) |
|---:|---:|---:|---:|---:|---:|
| 0.640 | 0.447 | 0.850 | 0.586 | 0.809 | 108 |

Confusion matrix: TN 77, FP 63, FN 9, TP 51. Lowering the threshold catches 85 % of defaulters (vs 35 % at 0.5) at the price of rejecting more good applicants — the trade-off the dataset's cost matrix asks for.

**Top global SHAP features**: checking account balance, loan duration, savings balance, credit history, credit amount.

**Fairness (test set, threshold 0.27)**

| Attribute | Approval rates | DP difference | EO difference | Status |
|---|---|---:|---:|---|
| Sex | female 41.0 % (n=61), male 43.9 % (n=139) | 0.029 | 0.150 | NEEDS ATTENTION |
| Age | ≥ 25: 47.8 % (n=161), < 25: 23.1 % (n=39) | 0.248 | 0.292 | CONCERNING |
| Foreign worker | yes 42.3 % (n=194), no 66.7 % (n=6) | 0.244 | 0.259 | CONCERNING (n=6, unreliable) |

The model disadvantages applicants under 25, even though part of the gap reflects the data (only 51 % of under-25 test applicants were actually good vs 75 % of older ones). Women are wrongly rejected more often (FPR 51 % vs 42 %) even though sex is not a model input — evidence of proxy effects. These are reported as-is; no mitigation was applied.

## 19. Limitations

- **Small dataset** (1,000 rows; 200 in test) → wide uncertainty on every metric, especially per group.
- **Not representative**: one German bank, 1970s–90s, Deutsche Mark amounts. The model would not generalise to today's or any Indian lending population.
- **Fairness**: sex is derived from a combined marital-status field ("female single" never appears); only 37 non-foreign workers exist in the whole dataset; removing sensitive features does not remove proxies; group metrics can conflict.
- **SHAP** explains the model, not causality, and depends on the background sample.
- **Stability** test is local and heuristic (random noise on two features), not a formal robustness guarantee.
- **Moderate accuracy** (ROC-AUC ≈ 0.81) — typical for this dataset.
- No authentication: this is a local demo.

## 20. Future work

- Fairness mitigation (Fairlearn `ThresholdOptimizer` or `ExponentiatedGradient`) with a before/after comparison.
- Probability calibration and confidence intervals (bootstrap) for metrics.
- Counterfactual explanations ("what would change the decision?").
- A larger, modern dataset (e.g. Home Credit) and drift monitoring.
- PDF report export for each assessment.
