"""Models, metrics, hyperparameter tuning and bias/variance diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fraud_metrics(y, score, amount=None, alert_rate: float = 0.005) -> dict:
    """PR-AUC, ROC-AUC, and what an operations team sees at a fixed review budget
    (top `alert_rate` of transactions sent to analysts)."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(len(y) * alert_rate)))
    top = np.argsort(-score)[:k]
    out = {
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        f"precision_at_{alert_rate:.1%}": round(float(y[top].mean()), 4),
        f"recall_at_{alert_rate:.1%}": round(float(y[top].sum() / max(y.sum(), 1)), 4),
        "positives": int(y.sum()), "n": int(len(y)),
    }
    if amount is not None:
        amount = np.asarray(amount)
        out[f"dollar_recall_at_{alert_rate:.1%}"] = round(float((amount[top] * y[top]).sum() / max((amount * y).sum(), 1e-9)), 4)
    return out


def logistic_regression(C: float = 1.0):
    """L2-regularised logistic regression; C is the inverse regularisation strength."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def gradient_boosting(**params):
    defaults = dict(learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
                    max_iter=300, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                    class_weight="balanced", random_state=0)
    defaults.update(params)
    return HistGradientBoostingClassifier(**defaults)


def tune_gradient_boosting(X, y, n_iter: int = 12, seed: int = 0):
    """Randomised search over learning rate, tree size, leaf size and L2 penalty, with time-ordered
    folds (never validate on the past) and PR-AUC as the objective."""
    space = {
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 50, 100, 200],
        "l2_regularization": [0.0, 0.5, 1.0, 3.0],
    }
    search = RandomizedSearchCV(gradient_boosting(), space, n_iter=n_iter, scoring="average_precision",
                                cv=TimeSeriesSplit(n_splits=3), random_state=seed, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search.best_estimator_, {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}, \
        round(float(search.best_score_), 4)


def complexity_curve(X_tr, y_tr, X_va, y_va, leaf_nodes=(2, 4, 8, 16, 31, 63, 127), min_leaf: int = 5) -> list[dict]:
    """Train vs validation PR-AUC as model capacity grows (no early stopping, little regularisation),
    to locate the bias/variance trade-off."""
    rows = []
    for m in leaf_nodes:
        model = gradient_boosting(max_leaf_nodes=m, min_samples_leaf=min_leaf, l2_regularization=0.0,
                                  early_stopping=False, max_iter=200).fit(X_tr, y_tr)
        rows.append({"max_leaf_nodes": m,
                     "train_pr_auc": round(float(average_precision_score(y_tr, model.predict_proba(X_tr)[:, 1])), 4),
                     "val_pr_auc": round(float(average_precision_score(y_va, model.predict_proba(X_va)[:, 1])), 4)})
    return rows


def diagnose(train_score: float, val_score: float, target: float = 0.6, gap: float = 0.1) -> str:
    """Rule of thumb: low train score = underfitting (high bias); big train/val gap = overfitting
    (high variance)."""
    if train_score < target:
        return "underfitting (high bias): add features or capacity, reduce regularisation"
    if train_score - val_score > gap:
        return "overfitting (high variance): more regularisation, smaller trees, more data or early stopping"
    return "good fit"
