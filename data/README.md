# Dataset — German Credit Data (Statlog)

- **Source:** Prof. Dr. Hans Hofmann, Universität Hamburg; UCI Machine Learning Repository, "Statlog (German Credit Data)" (1994). Licence: CC BY 4.0.
- **File:** `german_credit_raw.csv` — the original coded `german.data` file in comma-separated form (1,000 rows, 21 columns, no header). Decoding happens in `backend/app/ml/preprocessing.py` using the official code book.
- **Target:** column 21 — `1` = good credit (700), `2` = bad credit (300). We model `is_bad = (target == 2)`.
- **Missing values:** none.

| # | Column | Type | Meaning | Used by model? |
|---|---|---|---|---|
| 1 | checking_status | categorical | Status of existing checking account (< 0 DM, 0–200 DM, ≥ 200 DM, none) | yes |
| 2 | duration | numeric | Loan duration in months | yes |
| 3 | credit_history | categorical | Past repayment behaviour | yes |
| 4 | purpose | categorical | Purpose of the loan (car, furniture, education, …) | yes |
| 5 | credit_amount | numeric | Credit amount (DM) | yes |
| 6 | savings_status | categorical | Savings account / bonds | yes |
| 7 | employment | categorical | Present employment since | yes |
| 8 | installment_rate | numeric (1–4) | Installment as % of disposable income | yes |
| 9 | personal_status_sex | categorical | Marital status and sex | **no — sensitive, fairness audit only** |
| 10 | other_debtors | categorical | Co-applicant / guarantor | yes |
| 11 | residence_since | numeric (1–4) | Present residence since | yes |
| 12 | property | categorical | Most valuable property | yes |
| 13 | age | numeric | Age in years | yes (also audited) |
| 14 | other_installment_plans | categorical | Other installment plans (bank, stores, none) | yes |
| 15 | housing | categorical | Rent / own / for free | yes |
| 16 | existing_credits | numeric | Number of existing credits at this bank | yes |
| 17 | job | categorical | Job category | yes |
| 18 | num_dependents | numeric (1–2) | People liable to provide maintenance for | yes |
| 19 | telephone | categorical | Telephone registered | yes |
| 20 | foreign_worker | categorical | Foreign worker | **no — sensitive, fairness audit only** |

Derived audit columns (never model inputs): `sex` (from column 9), `age_group` (< 25 / ≥ 25), `foreign_worker_group`.

Notes and caveats:
- Code `A95` ("female : single") never occurs, so single women cannot be distinguished.
- Only 37 applicants are not foreign workers.
- The dataset's documentation provides a cost matrix: misclassifying a bad applicant as good costs 5, the reverse costs 1. We use it to choose the decision threshold.
- Amounts are in Deutsche Mark, and the data are decades old.
