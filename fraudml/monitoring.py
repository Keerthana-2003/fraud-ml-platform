"""Production monitoring and retraining: feature and score drift (PSI), a retraining policy, and
champion/challenger promotion through a file-based model registry."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference data.
    < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant or binary feature: compare category frequencies
        vals = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == v).mean() for v in vals])
        a = np.array([(actual == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, top: int = 5) -> dict:
    scores = {c: round(psi(reference[c], current[c]), 4) for c in reference.columns}
    worst = sorted(scores.items(), key=lambda kv: -kv[1])[:top]
    return {"max_psi": worst[0][1] if worst else 0.0, "top_features": dict(worst),
            "features_over_0.2": sum(v > 0.2 for v in scores.values())}


@dataclass
class RetrainPolicy:
    feature_psi: float = 0.2      # retrain if any important feature drifts this much
    score_psi: float = 0.1        # or the model's score distribution shifts
    precision_drop: float = 0.10  # or precision of analyst-reviewed alerts drops this much (labels for
                                  # unreviewed traffic arrive weeks late, so PR-AUC is not measurable yet)
    min_gain_to_promote: float = 0.01

    def should_retrain(self, max_feature_psi: float, score_psi: float, precision_drop: float | None) -> tuple[bool, list[str]]:
        reasons = []
        if max_feature_psi > self.feature_psi:
            reasons.append(f"feature PSI {max_feature_psi:.2f} > {self.feature_psi}")
        if score_psi > self.score_psi:
            reasons.append(f"score PSI {score_psi:.2f} > {self.score_psi}")
        if precision_drop is not None and precision_drop > self.precision_drop:
            reasons.append(f"reviewed-alert precision dropped {precision_drop:.2f} > {self.precision_drop}")
        return bool(reasons), reasons

    def promote(self, champion_pr_auc: float, challenger_pr_auc: float) -> bool:
        return challenger_pr_auc - champion_pr_auc >= self.min_gain_to_promote


@dataclass
class ModelRecord:
    version: str
    stage: str               # champion | challenger | archived
    trained_on: str          # data window description
    features_sha: str
    metrics: dict
    created_at: float


class ModelRegistry:
    """Minimal registry: versioned artifacts + metadata on disk, with stage transitions logged."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {"models": [], "events": []}

    def _save(self):
        self.index_path.write_text(json.dumps(self.index, indent=2))

    def register(self, model, features: list[str], trained_on: str, metrics: dict, stage: str = "challenger") -> ModelRecord:
        version = f"v{len(self.index['models']) + 1}"
        joblib.dump(model, self.root / f"{version}.joblib")
        rec = ModelRecord(version, stage, trained_on, hashlib.sha256(",".join(features).encode()).hexdigest()[:12],
                          metrics, time.time())
        self.index["models"].append(asdict(rec))
        self.index["events"].append({"event": "registered", "version": version, "stage": stage, "at": rec.created_at})
        self._save()
        return rec

    def champion(self) -> dict | None:
        return next((m for m in reversed(self.index["models"]) if m["stage"] == "champion"), None)

    def promote(self, version: str, reason: str) -> None:
        for m in self.index["models"]:
            if m["stage"] == "champion":
                m["stage"] = "archived"
            if m["version"] == version:
                m["stage"] = "champion"
        self.index["events"].append({"event": "promoted", "version": version, "reason": reason, "at": time.time()})
        self._save()

    def load(self, version: str):
        return joblib.load(self.root / f"{version}.joblib")
