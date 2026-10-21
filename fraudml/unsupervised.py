"""Unsupervised learning on customer behaviour: PCA (dimensionality reduction) and k-means clustering
(segmentation), plus a regression model of a customer's expected spend whose residual becomes a
fraud feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import CATEGORIES


class BehaviourSegments:
    """Standardise profiles -> PCA -> k-means with k picked by silhouette score."""

    def __init__(self, n_components: int = 5, k_range=range(2, 9), seed: int = 0):
        self.n_components, self.k_range, self.seed = n_components, k_range, seed

    def fit(self, profiles: pd.DataFrame) -> "BehaviourSegments":
        self.columns = list(profiles.columns)
        self.scaler = StandardScaler().fit(profiles)
        Z = self.scaler.transform(profiles)
        self.pca = PCA(n_components=self.n_components, random_state=self.seed).fit(Z)
        P = self.pca.transform(Z)
        self.silhouettes = {}
        for k in self.k_range:
            labels = KMeans(k, n_init=10, random_state=self.seed).fit_predict(P)
            self.silhouettes[k] = round(float(silhouette_score(P, labels)), 4)
        self.k = max(self.silhouettes, key=self.silhouettes.get)
        self.kmeans = KMeans(self.k, n_init=10, random_state=self.seed).fit(P)
        return self

    def transform(self, profiles: pd.DataFrame) -> pd.DataFrame:
        P = self.pca.transform(self.scaler.transform(profiles[self.columns]))
        out = pd.DataFrame(P, index=profiles.index, columns=[f"pc{i + 1}" for i in range(P.shape[1])])
        seg = self.kmeans.predict(P)
        for s in range(self.k):
            out[f"segment_{s}"] = (seg == s).astype(float)
        out["segment"] = seg
        return out

    def report(self, true_segments: pd.Series | None = None) -> dict:
        r = {"pca_explained_variance": [round(float(v), 4) for v in self.pca.explained_variance_ratio_],
             "pca_total_explained": round(float(self.pca.explained_variance_ratio_.sum()), 4),
             "silhouette_by_k": self.silhouettes, "chosen_k": int(self.k)}
        return r

    def agreement(self, profiles: pd.DataFrame, true_segments: pd.Series) -> float:
        pred = self.transform(profiles)["segment"]
        return round(float(adjusted_rand_score(true_segments.loc[pred.index], pred)), 4)


def expected_amount_features(df: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    """Inputs for predicting log(amount) from context only (no amount-derived columns)."""
    hours = (df["ts"] % 86400) / 3600
    X = pd.DataFrame({"hour_sin": np.sin(hours / 24 * 2 * np.pi), "hour_cos": np.cos(hours / 24 * 2 * np.pi),
                      "is_online": df["is_online"].astype(float).to_numpy()}, index=df.index)
    for c in CATEGORIES:
        X[f"cat_{c}"] = (df["category"] == c).astype(float).to_numpy()
    return X.join(cust.reindex(df["customer_id"]).set_axis(df.index).fillna(0))


class ExpectedAmountModel:
    """Regression: predict the log amount a customer would normally spend in this context.
    The residual (actual - expected) is a strong, interpretable fraud signal."""

    def fit(self, X: pd.DataFrame, log_amount: np.ndarray) -> "ExpectedAmountModel":
        self.columns = list(X.columns)
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0).fit(X, log_amount)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.columns])

    def evaluate(self, X: pd.DataFrame, log_amount: np.ndarray) -> dict:
        p = self.predict(X)
        return {"mae_log_amount": round(float(mean_absolute_error(log_amount, p)), 4),
                "r2": round(float(r2_score(log_amount, p)), 4),
                "median_abs_pct_error": round(float(np.median(np.abs(np.expm1(p) - np.expm1(log_amount)) / np.expm1(log_amount))), 4)}
