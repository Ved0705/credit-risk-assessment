"""
Dataset loading, decoding and preprocessing for the German Credit dataset
(Statlog, UCI Machine Learning Repository, Prof. Hans Hofmann, 1994).

The raw file uses symbolic codes such as "A11" or "A43". We decode them into
readable values using the official UCI documentation so that the API, the UI
and the explanations all speak in human terms.

Key design decisions (also documented in docs/methodology.md):
  * Target: original value 1 = good credit, 2 = bad credit.
    We model `is_bad` (1 = HIGH RISK / likely default, 0 = LOW RISK).
  * `personal_status_sex` and `foreign_worker` are SENSITIVE attributes.
    They are NOT given to the model ("fairness through unawareness"), but they
    are kept aside so that fairness can be measured on the test set.
  * `age` is also sensitive but is a standard and legally used credit factor
    in many jurisdictions; it is kept as a model feature AND audited for fairness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "target",
]

# Official UCI code book (german.doc)
CODEBOOK: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 DM", "A12": "0 - 200 DM",
        "A13": ">= 200 DM / salary assigned", "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits / all paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "others",
    },
    "savings_status": {
        "A61": "< 100 DM", "A62": "100 - 500 DM", "A63": "500 - 1000 DM",
        "A64": ">= 1000 DM", "A65": "unknown / no savings account",
    },
    "employment": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1 - 4 years",
        "A74": "4 - 7 years", "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male : divorced/separated",
        "A92": "female : divorced/separated/married",
        "A93": "male : single", "A94": "male : married/widowed",
        "A95": "female : single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other property", "A124": "unknown / no property",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {"A191": "none", "A192": "yes, registered"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]
CATEGORICAL_FEATURES = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Kept only for fairness auditing, never fed into the model
SENSITIVE_COLUMNS = ["personal_status_sex", "foreign_worker"]
TARGET = "is_bad"

# Valid ranges for numeric inputs (from the dataset, slightly widened)
NUMERIC_RANGES = {
    "duration": (4, 72, "Loan duration in months"),
    "credit_amount": (250, 20000, "Requested credit amount (Deutsche Mark)"),
    "installment_rate": (1, 4, "Installment rate as % of disposable income (1 = lowest, 4 = highest)"),
    "residence_since": (1, 4, "Years at present residence (1 = <1 yr ... 4 = >=4 yrs)"),
    "age": (18, 75, "Applicant age in years"),
    "existing_credits": (1, 4, "Number of existing credits at this bank"),
    "num_dependents": (1, 2, "Number of people the applicant must financially support"),
}

FEATURE_LABELS = {
    "checking_status": "Checking account balance",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)",
    "savings_status": "Savings balance",
    "employment": "Employment duration",
    "installment_rate": "Installment rate (% of income)",
    "other_debtors": "Other debtors / guarantors",
    "residence_since": "Years at residence",
    "property": "Most valuable property",
    "age": "Age",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at bank",
    "job": "Job type",
    "num_dependents": "Number of dependents",
    "telephone": "Telephone",
}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the coded UCI file (comma separated, no header)."""
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError("Unexpected number of columns in raw dataset")
    return df


def decode(df: pd.DataFrame) -> pd.DataFrame:
    """Translate symbolic codes into readable categories and build the target."""
    out = df.copy()
    for col, mapping in CODEBOOK.items():
        unknown = set(out[col].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unknown codes in {col}: {unknown}")
        out[col] = out[col].map(mapping)
    out[TARGET] = (out["target"] == 2).astype(int)
    return out.drop(columns=["target"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning + missing-value handling.

    The German Credit dataset has no missing values, but we still check so the
    pipeline is robust. Duplicates are dropped. Numeric columns are coerced.
    """
    out = df.drop_duplicates().copy()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # numeric -> median, categorical -> mode (only if anything is missing)
    for col in NUMERIC_FEATURES:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    for col in CATEGORICAL_FEATURES + SENSITIVE_COLUMNS:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode()[0])
    return out


def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive human-readable group columns used only for the fairness audit."""
    out = df.copy()
    out["sex"] = np.where(out["personal_status_sex"].str.startswith("female"), "female", "male")
    out["age_group"] = np.where(out["age"] < 25, "under 25", "25 and over")
    out["foreign_worker_group"] = np.where(out["foreign_worker"] == "yes", "foreign worker", "not foreign worker")
    return out


def load_dataset(path: str | Path) -> pd.DataFrame:
    return add_audit_columns(clean(decode(load_raw(path))))


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """ColumnTransformer: one-hot for categoricals, standard scaling for numerics.

    Tree models don't need scaling, but scaling is harmless for them and keeps
    one shared, reproducible preprocessing definition.
    """
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def categorical_options() -> dict[str, list[str]]:
    return {c: list(CODEBOOK[c].values()) for c in CATEGORICAL_FEATURES}
