"""Train the LSTM sequence model and compare it with gradient boosting on the same time split.

  python scripts/run_lstm.py            # full data (needs PyTorch; a few minutes on CPU)
  python scripts/run_lstm.py --small
Writes results/lstm_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraudml.data import GeneratorConfig, generate  # noqa: E402
from fraudml.distributed import build_features_distributed  # noqa: E402
from fraudml.modeling import fraud_metrics, gradient_boosting  # noqa: E402
from fraudml.sequence import LSTMScorer, build_sequences  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    df = generate(GeneratorConfig(n_customers=400 if args.small else 2000))
    X, _ = build_features_distributed(df)
    y, day, amt = df["is_fraud"].to_numpy(), df["day"].to_numpy(), df["amount"].to_numpy()
    tr, te = day < 60, day >= 75
    seqs = build_sequences(df, X, window=args.window)
    t0 = time.time()
    lstm = LSTMScorer(n_features=X.shape[1]).fit(seqs[tr], y[tr], epochs=args.epochs, X_val=seqs[te], y_val=y[te])
    train_s = time.time() - t0
    hgb = gradient_boosting().fit(X[tr], y[tr])
    s_lstm, s_hgb = lstm.predict_proba(seqs[te]), hgb.predict_proba(X[te])[:, 1]
    res = {"window": args.window, "epochs": args.epochs, "train_rows": int(tr.sum()), "train_seconds": round(train_s, 1),
           "history": lstm.history,
           "lstm": fraud_metrics(y[te], s_lstm, amt[te]),
           "gradient_boosting": fraud_metrics(y[te], s_hgb, amt[te]),
           "ensemble_mean_rank": fraud_metrics(y[te], (np.argsort(np.argsort(s_lstm)) + np.argsort(np.argsort(s_hgb))) / 2, amt[te])}
    Path("results").mkdir(exist_ok=True)
    Path("results/lstm_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"::notice title=LSTM results::lstm_pr_auc={res['lstm']['pr_auc']} hgb_pr_auc={res['gradient_boosting']['pr_auc']} "
          f"ensemble_pr_auc={res['ensemble_mean_rank']['pr_auc']} history={res['history']} train_s={res['train_seconds']}")


if __name__ == "__main__":
    main()
