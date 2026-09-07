from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.model import FraudMLP


def load_predictor(model_path: Path) -> tuple[dict[str, Any], FraudMLP]:
    """Load the saved model together with its feature contract and scaler."""
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    model = FraudMLP(len(artifact["feature_names"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return artifact, model


def score_feature_rows(
    artifact: dict[str, Any], model: FraudMLP, rows: list[dict[str, float | int]]
) -> np.ndarray:
    """Return one fraud probability per feature row using the saved contract."""
    feature_names = artifact["feature_names"]
    missing = sorted({name for name in feature_names for row in rows if name not in row})
    if missing:
        raise ValueError(f"Input is missing feature columns: {missing}")
    features = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=float)
    standardized = (features - np.asarray(artifact["scaler_mean"])) / np.asarray(artifact["scaler_scale"])
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(standardized, dtype=torch.float32))).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a CSV with a trained fraud model.")
    parser.add_argument("--model", type=Path, default=Path("models/fraud_mlp.pt"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    artifact, model = load_predictor(args.model)
    feature_names = artifact["feature_names"]
    frame = pd.read_csv(args.input)
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing feature columns: {missing}")
    probabilities = score_feature_rows(artifact, model, frame[feature_names].to_dict(orient="records"))
    output = frame.copy()
    output["fraud_probability"] = probabilities
    output["review_recommended"] = (probabilities >= artifact["threshold"]).astype(int)
    output.to_csv(args.output, index=False)
    print(f"Scored {len(output)} rows; wrote {args.output}")


if __name__ == "__main__":
    main()
