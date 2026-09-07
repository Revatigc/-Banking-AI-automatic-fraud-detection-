"""Measure the generator-aware (Bayes-optimal) reference on the demo split.

The benchmark ranks transactions with the latent probability used to generate
their labels. It is an empirical ceiling/reference for this synthetic setup,
not a real-world performance estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data import TARGET_COLUMN, make_demo_data

FLOORS = (0.2, 0.4, 0.5, 0.6, 0.75, 0.8)
SEED = 7


def choose_threshold(labels: np.ndarray, scores: np.ndarray, floor: float) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(precision[:-1] >= floor)
    return float(thresholds[eligible[np.argmax(recall[eligible])]]) if len(eligible) else 0.5


def main() -> None:
    frame = make_demo_data(seed=SEED, include_oracle=True)
    labels = frame[TARGET_COLUMN].to_numpy()
    scores = frame.pop("oracle_fraud_probability").to_numpy()
    indices = np.arange(len(frame))
    _, other_indices, _, other_labels = train_test_split(
        indices, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    valid_indices, test_indices, valid_labels, test_labels = train_test_split(
        other_indices, other_labels, test_size=0.50, stratify=other_labels, random_state=SEED
    )
    results = []
    for floor in FLOORS:
        threshold = choose_threshold(valid_labels, scores[valid_indices], floor)
        prediction = (scores[test_indices] >= threshold).astype(int)
        results.append(
            {
                "minimum_precision": floor,
                "average_precision": round(float(average_precision_score(test_labels, scores[test_indices])), 4),
                "precision": round(float(precision_score(test_labels, prediction, zero_division=0)), 4),
                "recall": round(float(recall_score(test_labels, prediction, zero_division=0)), 4),
                "threshold": round(threshold, 4),
            }
        )
    output = {
        "description": "Generator-aware Bayes-optimal reference on the same synthetic validation/test split.",
        "seed": SEED,
        "test_rows": int(len(test_indices)),
        "fraud_rate_test": round(float(test_labels.mean()), 4),
        "operating_points": results,
    }
    destination = Path("models/demo_oracle.metrics.json")
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved oracle reference to {destination}")


if __name__ == "__main__":
    main()
