"""
Prediction stability / robustness test.

1. Random perturbation: create N copies of the applicant where the continuous,
   non-sensitive features (loan duration, credit amount) are multiplied by
   (1 + Normal(0, noise_level)), then re-predict and count how many decisions stay the same.
2. One-feature-at-a-time sensitivity: nudge each non-sensitive numeric feature
   slightly (±noise / ±2·noise for continuous, ±1 step for small integer scales)
   and record how much the probability moves and whether the decision flips.

Age is deliberately NOT perturbed: it is a sensitive attribute, and "what if the
applicant were older" is a fairness question, not a robustness question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ml.model_loader import get_model
from ..ml.preprocessing import FEATURE_LABELS, NUMERIC_RANGES
from .prediction_service import classify, predict_probability

CONTINUOUS = ["duration", "credit_amount"]
ORDINAL = ["installment_rate", "residence_since", "existing_credits", "num_dependents"]


def _clip(feat: str, v: float) -> float:
    lo, hi, _ = NUMERIC_RANGES[feat]
    return float(min(max(round(v), lo), hi))


def analyse(applicant: dict, n_samples: int = 20, noise_level: float = 0.05, seed: int = 42) -> dict:
    bundle = get_model()
    t = bundle.threshold
    base_p = predict_probability(pd.DataFrame([applicant]))[0]
    base_pred, _ = classify(base_p, t)

    # --- 1. joint random perturbations ---
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        r = dict(applicant)
        for f in CONTINUOUS:
            r[f] = _clip(f, applicant[f] * (1 + rng.normal(0, noise_level)))
        rows.append(r)
    probs = np.array(predict_probability(pd.DataFrame(rows)))
    same = int(sum(classify(p, t)[0] == base_pred for p in probs))
    score = same / n_samples

    # --- 2. one-at-a-time sensitivity ---
    sens = []
    for f in CONTINUOUS + ORDINAL:
        orig = float(applicant[f])
        if f in CONTINUOUS:
            cands = [orig * (1 + k * noise_level) for k in (-2, -1, 1, 2)]
        else:
            cands = [orig - 1, orig + 1]
        values = sorted({_clip(f, v) for v in cands} - {orig})
        if not values:
            continue
        variants = [dict(applicant, **{f: v}) for v in values]
        ps = predict_probability(pd.DataFrame(variants))
        sens.append({
            "feature": f, "label": FEATURE_LABELS[f], "original_value": orig,
            "tested_values": values, "probabilities": [round(p, 4) for p in ps],
            "max_probability_change": round(float(max(abs(p - base_p) for p in ps)), 4),
            "flips_decision": any(classify(p, t)[0] != base_pred for p in ps),
        })
    sens.sort(key=lambda s: (s["flips_decision"], s["max_probability_change"]), reverse=True)

    status = "STABLE" if score >= 0.9 else "MODERATELY STABLE" if score >= 0.7 else "UNSTABLE"
    flippers = [s["label"] for s in sens if s["flips_decision"]]
    margin = abs(base_p - t)
    interpretation = (
        f"{same} of {n_samples} slightly modified versions of this applicant (±{noise_level:.0%} noise on "
        f"loan duration and amount) received the same decision ({score:.0%}). "
        f"The applicant's default probability ({base_p:.0%}) is {margin:.0%} away from the decision threshold ({t:.0%}). "
    )
    interpretation += (
        f"Small changes to {', '.join(flippers)} can flip the decision, so this case is borderline and deserves human review."
        if flippers else "No single small change flipped the decision."
    )
    return {
        "original_prediction": base_pred,
        "original_probability": round(base_p, 4),
        "n_samples": n_samples, "noise_level": noise_level,
        "same_prediction": same, "stability_score": round(score, 4), "status": status,
        "probability_mean": round(float(probs.mean()), 4),
        "probability_std": round(float(probs.std()), 4),
        "probability_min": round(float(probs.min()), 4),
        "probability_max": round(float(probs.max()), 4),
        "sensitive_features": sens,
        "perturbed_features": CONTINUOUS,
        "interpretation": interpretation,
    }
