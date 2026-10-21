"""Sequence model: an LSTM over each customer's recent transactions (PyTorch).

Tabular features summarise history with hand-built aggregates; the LSTM instead reads the raw
sequence of the last `window` transactions (features + time gaps) and scores the latest one, so it
can learn patterns such as "many tiny online charges, then a large one" directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences(df: pd.DataFrame, feats: pd.DataFrame, window: int = 10) -> np.ndarray:
    """(n_txns, window, n_features) float32: each row is the customer's previous window-1
    transactions plus the current one, left-padded with zeros. Row order matches df."""
    X = feats.to_numpy(np.float32)
    seqs = np.zeros((len(df), window, X.shape[1]), np.float32)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for _, g in df.sort_values("ts", kind="stable").groupby("customer_id", sort=False):
        rows = pos.loc[g.index].to_numpy()
        for j, r in enumerate(rows):
            hist = rows[max(0, j - window + 1): j + 1]
            seqs[r, window - len(hist):] = X[hist]
    return seqs


class LSTMScorer:
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1, lr: float = 2e-3, seed: int = 0,
                 device: str | None = None):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(n_features)
                self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                    dropout=0.1 if layers > 1 else 0.0)
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

            def forward(self, x):
                out, _ = self.lstm(self.norm(x))
                return self.head(out[:, -1]).squeeze(-1)

        self.net = Net().to(self.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)  # L2-style regularisation
        self.history: list[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 3, batch: int = 512,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "LSTMScorer":
        torch = self.torch
        pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=self.device).clamp(max=200)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # weighted loss for rare positives
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            self.net.train()
            idx = rng.permutation(len(X))
            total = 0.0
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                xb, yb = Xt[b].to(self.device), yt[b].to(self.device)
                self.opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                total += float(loss) * len(b)
            entry = {"epoch": ep + 1, "train_loss": round(total / len(X), 4)}
            if X_val is not None:
                from sklearn.metrics import average_precision_score

                entry["val_pr_auc"] = round(float(average_precision_score(y_val, self.predict_proba(X_val))), 4)
            self.history.append(entry)
        return self

    def predict_proba(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), batch):
                out.append(torch.sigmoid(self.net(torch.from_numpy(X[s:s + batch]).to(self.device))).cpu().numpy())
        return np.concatenate(out)
