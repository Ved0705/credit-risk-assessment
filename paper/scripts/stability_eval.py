"""Reproduce the stability results reported in Section 17.4 (Table V).

Runs the system's stability analysis on every applicant of the held-out test set.
Run from the project's backend/ folder (after training):
    python ../paper/scripts/stability_eval.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.getcwd())  # run from backend/ so "app" is importable

from app.ml.model_loader import get_model
from app.ml.preprocessing import MODEL_FEATURES
from app.services.stability_service import analyse

bundle = get_model()
test = bundle.test_set[MODEL_FEATURES]
scores, probs, flips = [], [], Counter()
any_flip = 0
for _, row in test.iterrows():
    r = analyse(row.to_dict(), n_samples=20, noise_level=0.05, seed=42)
    scores.append(r["stability_score"])
    probs.append(r["original_probability"])
    flipped = [f["feature"] for f in r["sensitive_features"] if f["flips_decision"]]
    any_flip += bool(flipped)
    flips.update(flipped)

s, p = np.array(scores), np.array(probs)
near = np.abs(p - bundle.threshold) < 0.05
n = len(s)
print(f"applicants: {n}, threshold: {bundle.threshold}")
print(f"mean stability: {s.mean():.3f}")
print(f"S = 1: {(s == 1).sum()}/{n} ({(s == 1).mean():.1%})")
print(f"S >= 0.9: {(s >= 0.9).sum()}/{n} ({(s >= 0.9).mean():.1%})")
print(f"lowest S: {s.min():.2f}")
print(f"one-feature change flips decision: {any_flip}/{n} ({any_flip / n:.1%})")
print(f"mean S near threshold (n={near.sum()}): {s[near].mean():.3f}; others (n={(~near).sum()}): {s[~near].mean():.3f}")
print("features involved in flips:", dict(flips))
