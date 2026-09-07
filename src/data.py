from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "is_fraud"


def make_demo_data(rows: int = 8_000, seed: int = 7, include_oracle: bool = False) -> pd.DataFrame:
    """Create a synthetic, non-linear fraud dataset for reproducibility checks.

    ``include_oracle`` exposes the generator's latent probability only for the
    oracle benchmark script. It must never be passed to model training.
    """
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=3.6, sigma=1.0, size=rows)
    hour = rng.integers(0, 24, size=rows)
    account_age_days = rng.gamma(shape=2.0, scale=240, size=rows)
    transactions_24h = rng.poisson(lam=2.0, size=rows)
    distance_km = rng.exponential(scale=18, size=rows)
    card_present = rng.binomial(1, 0.7, size=rows)

    # Interaction-driven patterns approximate rules an analyst might surface:
    # an overnight burst from a young account, or a remote high-value
    # card-not-present purchase. These are intentionally not extra columns;
    # the model must learn the relationships from the six observable fields.
    overnight = hour < 6
    young_account = account_age_days < 365
    rapid_velocity = transactions_24h >= 3
    remote = distance_km >= 25
    high_value = amount >= 100
    card_not_present = card_present == 0
    high_signal_a = overnight & young_account & rapid_velocity
    high_signal_b = high_value & remote & card_not_present
    elevated_pattern = (overnight & rapid_velocity) | (young_account & card_not_present & (transactions_24h >= 3))

    risk = (
        -6.3
        + 0.004 * np.minimum(amount, 800)
        + 0.045 * np.minimum(transactions_24h, 12)
        + 0.009 * np.minimum(distance_km, 180)
        + 0.35 * card_not_present
        + 0.4 * overnight
        + 0.3 * young_account
        + 1.2 * elevated_pattern
    )
    probability = 1 / (1 + np.exp(-risk))
    probability = np.where(elevated_pattern, np.maximum(probability, 0.68), probability)
    probability = np.where(high_signal_a | high_signal_b, 0.98, probability)
    is_fraud = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "account_age_days": account_age_days,
            "transactions_24h": transactions_24h,
            "distance_km": distance_km,
            "card_present": card_present,
            TARGET_COLUMN: is_fraud,
        }
    )
    if include_oracle:
        frame["oracle_fraud_probability"] = probability
    return frame


def load_data(path: Path, demo: bool) -> pd.DataFrame:
    if demo:
        return make_demo_data()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Use --demo for a smoke test.")
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"CSV must contain a '{TARGET_COLUMN}' column.")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"'{TARGET_COLUMN}' must contain both 0 and 1 labels.")
    return frame


def numeric_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"CSV must contain a '{TARGET_COLUMN}' column.")
    if frame[TARGET_COLUMN].isnull().any():
        raise ValueError(f"'{TARGET_COLUMN}' cannot contain missing values.")
    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"'{TARGET_COLUMN}' must contain only 0 and 1 labels.")
    raw_features = frame.drop(columns=[TARGET_COLUMN])
    non_numeric = raw_features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"All feature columns must be numeric; found: {non_numeric}")
    features = raw_features.copy()
    if features.empty:
        raise ValueError("No numeric feature columns found after removing the target.")
    if features.isnull().any().any():
        raise ValueError("Missing values found. Impute them upstream to keep preprocessing explicit.")
    labels = frame[TARGET_COLUMN].astype(int).to_numpy()
    return features, labels
