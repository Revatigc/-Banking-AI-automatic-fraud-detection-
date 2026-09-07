from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_data, numeric_feature_frame
from src.model import FraudMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, minimum_precision: float) -> float:
    """Maximize recall subject to a precision guardrail on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= minimum_precision)
    if len(eligible) == 0:
        return 0.5
    return float(thresholds[eligible[np.argmax(recall[eligible])]])


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def predict(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch tabular fraud classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/transactions.csv"))
    parser.add_argument("--demo", action="store_true", help="Use synthetic data only to smoke-test the pipeline.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minimum-precision", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("models/fraud_mlp.pt"))
    args = parser.parse_args()
    if not 0 < args.minimum_precision <= 1:
        raise ValueError("--minimum-precision must be in (0, 1].")

    set_seed(args.seed)
    features, labels = numeric_feature_frame(load_data(args.data, args.demo))
    x_train, x_other, y_train, y_other = train_test_split(
        features.to_numpy(), labels, test_size=0.30, stratify=labels, random_state=args.seed
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_other, y_other, test_size=0.50, stratify=y_other, random_state=args.seed
    )
    scaler = StandardScaler().fit(x_train)
    x_train, x_valid, x_test = (scaler.transform(x) for x in (x_train, x_valid, x_test))

    model = FraudMLP(x_train.shape[1])
    positive_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, loss_fn)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:02d} train_loss={loss:.4f}")

    validation_probabilities = predict(model, x_valid)
    threshold = choose_threshold(y_valid, validation_probabilities, args.minimum_precision)
    test_probabilities = predict(model, x_test)
    predictions = (test_probabilities >= threshold).astype(int)
    metrics = {
        "average_precision": round(float(average_precision_score(y_test, test_probabilities)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "threshold": round(threshold, 4),
        "test_rows": int(len(y_test)),
        "fraud_rate_test": round(float(y_test.mean()), 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": list(features.columns),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "threshold": threshold,
        },
        args.output,
    )
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.output} and metrics to {metrics_path}")


if __name__ == "__main__":
    main()
