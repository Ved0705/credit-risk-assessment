"""
Train, tune, compare and save credit-risk models.

Run from the backend/ folder:
    python -m app.ml.train

Steps
  1. Load + decode + clean the German Credit dataset
  2. Stratified 80/20 train/test split (test set is never used for tuning)
  3. GridSearchCV (5-fold stratified, ROC-AUC) for Logistic Regression,
     Random Forest and XGBoost — preprocessing lives INSIDE the pipeline,
     so scaling/encoding is fitted on training folds only (no leakage)
  4. Select the model by cross-validated ROC-AUC on the training set; a simpler
     model wins if it is within 0.01 AUC of a more complex one
  5. Choose a decision threshold using out-of-fold training predictions and
     the dataset's 5:1 cost matrix
  6. Evaluate every model on the held-out test set, both at 0.5 and at EACH model's
     own cost-optimal threshold (also chosen from its out-of-fold training predictions),
     so models are compared the way they would actually be deployed
  7. Bootstrap 95 % confidence intervals for the selected model's test metrics
  8. Save the pipeline, metadata, test set and SHAP global importance
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ..config import settings
from .evaluate import COST_FN, COST_FP, best_cost_threshold, classification_metrics
from .explainer import ShapExplainer
from .uncertainty import N_BOOT, metric_cis
from .preprocessing import (
    CATEGORICAL_FEATURES, FEATURE_LABELS, MODEL_FEATURES, NUMERIC_FEATURES,
    NUMERIC_RANGES, SENSITIVE_COLUMNS, TARGET, build_preprocessor,
    categorical_options, load_dataset,
)

SEED = 42

# Ordered from simplest to most complex
CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {"classifier__C": [0.01, 0.1, 0.3, 1.0, 3.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {
            "classifier__n_estimators": [300],
            "classifier__max_depth": [4, 8, None],
            "classifier__min_samples_leaf": [1, 5],
        },
    ),
    "XGBoost": (
        XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=2, verbosity=0),
        {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__subsample": [0.8],
        },
    ),
}
SIMPLICITY_MARGIN = 0.01


def main() -> None:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(settings.data_path)
    print(f"Loaded {len(df)} rows; bad-credit rate = {df[TARGET].mean():.1%}")

    X = df[MODEL_FEATURES]
    y = df[TARGET]
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        proba_test = best.predict_proba(X_test)[:, 1]
        # each model gets its own cost-optimal threshold, from out-of-fold TRAINING predictions only
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        t_cost = best_cost_threshold(y_train, oof)
        results[name] = {
            "estimator": best,
            "cv_roc_auc": round(float(search.best_score_), 4),
            "best_params": {k.replace("classifier__", ""): v for k, v in search.best_params_.items()},
            "test_metrics_at_0_5": classification_metrics(y_test, proba_test, 0.5),
            "cost_threshold": t_cost,
            "test_metrics_at_cost_threshold": classification_metrics(y_test, proba_test, t_cost),
        }
        print(f"{name:20s} CV AUC={search.best_score_:.4f}  test AUC={results[name]['test_metrics_at_0_5']['roc_auc']:.4f}"
              f"  cost threshold={t_cost}  test cost={results[name]['test_metrics_at_cost_threshold']['misclassification_cost']}")

    # ---- model selection (training CV only) ----
    names = list(CANDIDATES)  # simplest first
    best_auc = max(r["cv_roc_auc"] for r in results.values())
    selected = next(n for n in names if results[n]["cv_roc_auc"] >= best_auc - SIMPLICITY_MARGIN)
    top = max(names, key=lambda n: results[n]["cv_roc_auc"])
    if selected == top:
        reason = f"{selected} had the highest 5-fold cross-validated ROC-AUC on the training set ({results[selected]['cv_roc_auc']})."
    else:
        reason = (f"{top} had the highest CV ROC-AUC ({results[top]['cv_roc_auc']}), but {selected} "
                  f"({results[selected]['cv_roc_auc']}) is within {SIMPLICITY_MARGIN} and is simpler and more "
                  f"interpretable, so it was preferred.")
    print("Selected:", selected, "-", reason)
    model = results[selected]["estimator"]

    # ---- threshold from out-of-fold training predictions (computed above) ----
    threshold = results[selected]["cost_threshold"]
    print("Cost-optimal threshold (train OOF):", threshold)

    proba_test = model.predict_proba(X_test)[:, 1]
    selected_metrics = classification_metrics(y_test, proba_test, threshold)
    selected_ci = metric_cis(y_test, proba_test, threshold)
    print("95% bootstrap CI for test ROC-AUC:", selected_ci["roc_auc"])

    # ---- SHAP global importance (test set) ----
    background = X_train.sample(100, random_state=SEED)
    explainer = ShapExplainer(model, background)
    global_importance = explainer.global_importance(X_test)

    # ---- save artifacts ----
    joblib.dump(model, settings.model_dir / "model.joblib")
    background.to_csv(settings.model_dir / "shap_background.csv", index=False)
    audit_cols = SENSITIVE_COLUMNS + ["sex", "age_group", "foreign_worker_group"]
    df.loc[idx_test, MODEL_FEATURES + audit_cols + [TARGET]].to_csv(settings.model_dir / "test_set.csv", index=False)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "German Credit (Statlog), UCI ML Repository",
            "rows": int(len(df)), "train_rows": int(len(idx_train)), "test_rows": int(len(idx_test)),
            "bad_credit_rate": round(float(y.mean()), 4),
        },
        "selected_model": selected,
        "selection_reason": reason,
        "decision_threshold": threshold,
        "threshold_reason": (f"Minimises the dataset's published cost matrix (missed bad applicant = {COST_FN}, "
                             f"wrongly rejected good applicant = {COST_FP}) on out-of-fold training predictions."),
        "selected_metrics": selected_metrics,
        "selected_metrics_ci": selected_ci,
        "ci_method": (f"95% percentile bootstrap over the {len(idx_test)} test applicants "
                      f"({N_BOOT} resamples, seed {SEED}); the model is not retrained."),
        # top-level metrics in each row are at the 0.5 threshold (kept for backwards compatibility);
        # `at_cost_threshold` shows each model at its own cost-optimal threshold
        "comparison": [
            {"model": n, "cv_roc_auc": r["cv_roc_auc"], "best_params": r["best_params"], **r["test_metrics_at_0_5"],
             "cost_threshold": r["cost_threshold"], "at_cost_threshold": r["test_metrics_at_cost_threshold"]}
            for n, r in results.items()
        ],
        "global_importance": global_importance,
        "shap_output_units": explainer.output_units,
        "features": {
            "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
            "excluded_sensitive": SENSITIVE_COLUMNS,
        },
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
    }
    with open(settings.model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(settings.model_dir / "feature_schema.json", "w") as f:
        json.dump({
            "numeric": {k: {"min": v[0], "max": v[1], "description": v[2]} for k, v in NUMERIC_RANGES.items()},
            "categorical": categorical_options(),
            "labels": FEATURE_LABELS,
        }, f, indent=2)
    print("Saved artifacts to", settings.model_dir)
    print(json.dumps(selected_metrics, indent=2))


if __name__ == "__main__":
    main()
