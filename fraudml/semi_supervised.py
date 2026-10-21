"""Semi-supervised learning for delayed and partial labels.

At training time only a reviewed sample of transactions has labels (and fraud labels arrive weeks
late through chargebacks). Self-training fits on the labelled sample, pseudo-labels the unlabelled
transactions it is most confident about, and refits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier

from .modeling import gradient_boosting


def labels_known_at(df: pd.DataFrame, cutoff_ts: float) -> np.ndarray:
    """Fraud label as known at `cutoff_ts`: frauds whose chargeback has not arrived look legitimate."""
    return ((df["is_fraud"] == 1) & (df["label_ts"] <= cutoff_ts)).astype(int).to_numpy()


def reviewed_sample(n: int, y_known: np.ndarray, review_rate: float, seed: int = 0) -> np.ndarray:
    """Mask of labelled rows: every known chargeback plus a random review sample of the rest."""
    rng = np.random.default_rng(seed)
    return (y_known == 1) | (rng.random(n) < review_rate)


def self_training(threshold: float = 0.95, max_iter: int = 5):
    base = gradient_boosting(class_weight=None, max_iter=200)
    return SelfTrainingClassifier(base, threshold=threshold, max_iter=max_iter)
