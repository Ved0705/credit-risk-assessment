# Methodology

## 1. Data preparation
1. Read the coded UCI file; decode every symbolic code with the official code book (unknown codes raise an error).
2. Build target `is_bad` (1 = bad credit, 30 % of rows).
3. Clean: drop duplicates (none found), coerce numeric types, impute numeric with median and categorical with mode (no missing values exist, but the step keeps the pipeline robust).
4. Separate sensitive columns (`personal_status_sex`, `foreign_worker`) from model features; derive audit groups `sex`, `age_group`, `foreign_worker_group`.

**Model features (18):** 7 numeric (duration, credit_amount, installment_rate, residence_since, age, existing_credits, num_dependents) and 11 categorical.

## 2. Preprocessing
`ColumnTransformer`: `StandardScaler` for numeric, `OneHotEncoder(handle_unknown="ignore")` for categorical → 55 model columns. It lives inside the scikit-learn `Pipeline`, so `GridSearchCV` refits it on each training fold (no leakage from validation/test data).

## 3. Split and tuning
- Stratified 80/20 split, `random_state=42` → 800 train / 200 test.
- 5-fold stratified CV on the training set, metric ROC-AUC.

| Model | Grid |
|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 0.3, 1, 3} |
| Random Forest | 300 trees, max_depth ∈ {4, 8, None}, min_samples_leaf ∈ {1, 5} |
| XGBoost | n_estimators ∈ {100, 300}, max_depth ∈ {2, 3, 4}, learning_rate ∈ {0.03, 0.1}, subsample 0.8 |

## 4. Model selection
Highest CV ROC-AUC wins, **unless** a simpler model is within 0.01 — then the simpler model is chosen (simplicity order LR < RF < XGB). ROC-AUC is used because it measures ranking quality independent of any threshold and is robust to the 70/30 class imbalance. The test set is never used for selection.

## 5. Decision threshold
Default 0.5 is not appropriate when errors have unequal costs. Using out-of-fold predictions on the training set (`cross_val_predict`), we scan thresholds 0.05–0.95 and choose the one minimising `5·FN + 1·FP` (cost matrix from the dataset documentation). Result: 0.27.

For a fair comparison, each candidate model gets its own cost-optimal threshold chosen the same way (out-of-fold training predictions), and is reported on the test set at that threshold as well as at 0.5. This is reporting only; selection still uses CV ROC-AUC.

Risk bands: LOW (p < threshold), MEDIUM (threshold ≤ p < 0.5 — flagged but borderline, human review), HIGH (p ≥ 0.5).

## 6. Why accuracy is insufficient
A model approving everyone scores 70 % accuracy and catches no defaulters. We therefore report precision, recall, F1, ROC-AUC, the confusion matrix and the business cost alongside accuracy.

## 7. Explainability (SHAP)
- Tree models: `shap.TreeExplainer(model, data=background, model_output="probability")` (interventional), background = 100 random training rows. Linear model: `shap.LinearExplainer`.
- One-hot column SHAP values are summed per original feature.
- Positive SHAP → towards HIGH RISK; negative → towards LOW RISK.
- Local accuracy check: base value + Σ SHAP ≈ predicted probability (test tolerance 0.01).
- Global importance: mean |SHAP| over the test set.

## 8. Fairness (Fairlearn)
Computed on the test set at the deployed threshold:
- Per group: count, approval rate (predicted LOW RISK), FPR, FNR, accuracy, actual good rate.
- Demographic parity difference/ratio: gap/ratio in approval rates.
- Equalized odds difference: max gap in TPR or FPR.
- Status from max(DPD, EOD): < 0.10 acceptable, < 0.20 needs attention, else concerning.
- Small groups (n < 30) are flagged as unreliable, and the attribute gets the status INSUFFICIENT DATA (no verdict; excluded from the overall status).
- DPD and EOD carry 95 % percentile-bootstrap intervals (1,000 test-set resamples, seed 42).
- The data were not altered to improve the metrics.

## 9. Stability
Random perturbation of duration and credit amount (relative Gaussian noise σ = 5 %, clipped, rounded, fixed seed) → share of unchanged decisions. Plus one-at-a-time sensitivity for each non-sensitive numeric feature. Age is excluded on purpose.

## 10. Uncertainty
The test set has 200 applicants, so headline metrics (accuracy, precision, recall, F1, ROC-AUC, cost) are reported with 95 % percentile-bootstrap confidence intervals (1,000 resamples of the test set, seed 42, model not retrained).

## 11. Reproducibility
Seeds fixed (42) for split, CV, models, SHAP background and stability. Results can still change slightly across library versions; `metadata.json` records the versions used, and `backend/requirements.txt` pins the exact versions behind the published results. XGBoost's numbers can differ slightly between machines (multi-threaded training); the selected Random Forest reproduces exactly.
