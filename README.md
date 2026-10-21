# Fraud ML Platform

The full machine learning lifecycle for card-fraud detection, built the way a production team
would run it: point-in-time features, distributed data preparation, model selection and tuning,
bias/variance diagnostics, unsupervised and semi-supervised learning, a sequence model,
a reinforcement-learning review policy, drift monitoring, retraining with a champion/challenger
registry, and responsible-AI reporting.

```
synthetic transactions ──► point-in-time features ──► (Ray | process pool) sharded by customer
                                  │
    ┌─────────────────────────────┼──────────────────────────────────────────┐
    ▼                             ▼                                          ▼
 PCA + k-means segments   regression: expected spend   supervised: logistic regression (L2),
 (customer behaviour)      residual = fraud signal      gradient boosting + time-series tuning
    └──────────────┬──────────────┘                     semi-supervised: self-training
                   ▼                                     sequence: LSTM over recent transactions
            champion model ──► weekly PSI drift ──► retrain policy ──► challenger ──► promote?
                   │                                                (model registry, guardrail)
                   ▼
      contextual bandit picks each segment's review rate · slice metrics · explanations · model card
```

## Results

From `python scripts/run_experiments.py` (full run, about 2 minutes on 2 CPU cores). Every number
below is in [`results/metrics.json`](results/metrics.json); the model card is
[`results/model_card.md`](results/model_card.md).

Data: 144,338 synthetic transactions, 2,000 customers, 90 days, 0.43% fraud, behaviour drift from day 60.
Time-based split: train days 0-49, validation 50-59, monitoring 60-74, final test 75-89.

| Experiment | Result |
|---|---|
| Logistic regression (L2) vs gradient boosting, validation PR-AUC | 0.092 vs 0.364 (default) → **0.403 tuned** |
| Hyperparameter search (time-series CV, PR-AUC objective) | best: 15 leaves, min 100 per leaf, learning rate 0.08, L2 0.5 |
| Bias / variance (train vs val PR-AUC by tree size) | 2 leaves 0.30 / 0.30 (underfit) → 16 leaves 1.00 / 0.50 (overfit); tuned model regularised |
| Regression: expected spend per customer and context | R² 0.36, MAE 0.55 log-dollars on legitimate validation spend |
| PCA + k-means customer segments | 5 components explain 60% of variance; silhouette score picked k = 2 |
| Feature ablation: + spend residual, PCA components, segments | validation PR-AUC 0.403 → **0.466** |
| Semi-supervised (10% reviewed, only 59 of 145 fraud chargebacks arrived) | supervised-only 0.355 → **self-training 0.417** |
| Champion on final post-drift window (days 75-89) | PR-AUC 0.457, ROC-AUC 0.965, precision 0.55 at a 0.5% review budget |
| Drift monitoring | weekly max feature PSI rose 0.16 → 0.23 → 0.30 after the shift; score PSI stayed below 0.08 |
| Retraining | triggered by PSI 0.23 > 0.2; challenger scored 0.426 vs champion 0.457, so the guardrail **kept the champion** |
| LSTM over the last 10 transactions (PyTorch, weighted BCE, 4 epochs; CI run) | post-drift test PR-AUC **0.634** vs 0.487 for default gradient boosting on the same split; rank-average ensemble **0.686** |
| Contextual bandit for review rates (2,000 rounds) | 3.6% more net fraud dollars caught than a fixed 1% review rate |
| Responsible AI | permutation importance, per-alert explanations, FPR by segment (max/min ratio 1.6) and by channel |

Findings worth discussing:
- Gradient boosting beats regularised logistic regression by 4x on PR-AUC because fraud depends on
  interactions (new merchant **and** far from home **and** large amount).
- Labels arrive late: by the review on day 75 only 8 of 150 post-drift frauds had chargebacks, so
  PR-AUC cannot be measured on fresh data. The retrain policy uses feature/score PSI and the
  precision of analyst-reviewed alerts instead.
- A retrained model is not automatically better. Training on recent data with immature labels teaches
  the model that unreported fraud is legitimate; the champion/challenger guardrail caught that.
- Online transactions get ~14x the false-positive rate of in-person ones at the same threshold; the
  model card flags this for review.

The LSTM row comes from the `lstm` job in CI (`.github/workflows/ci.yml`), which trains on the full
dataset and uploads `results/lstm_metrics.json` as an artifact; run it locally with
`python scripts/run_lstm.py`. The sequence model's gain suggests the order of recent transactions
(tiny charges followed by a large one) carries signal the hand-built aggregates miss.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # add ".[ray]" for Ray, ".[torch]" for the LSTM
pytest -q
python scripts/run_experiments.py       # writes results/
```

Run feature preparation on a Ray cluster with `RAY_ADDRESS=ray://<head>:10001`; the same sharding
by customer maps to Spark `groupBy("customer_id").applyInPandas`.

## Layout

```
fraudml/data.py            synthetic transactions: segments, fraud bursts, label delay, drift
fraudml/features.py        point-in-time behavioural features (no future leakage, tested)
fraudml/distributed.py     sharded feature preparation on Ray or a process pool
fraudml/modeling.py        metrics at a review budget, logistic regression, gradient boosting, tuning, bias/variance
fraudml/unsupervised.py    PCA + k-means segments, expected-spend regression
fraudml/semi_supervised.py label delay, reviewed samples, self-training
fraudml/sequence.py        LSTM over each customer's recent transactions (PyTorch)
fraudml/bandit.py          Thompson-sampling contextual bandit for review rates
fraudml/monitoring.py      PSI drift, retrain policy, champion/challenger model registry
fraudml/responsible.py     permutation importance, per-alert explanations, slice metrics, model card
scripts/                   run_experiments.py, run_lstm.py
```

## Limitations

The data is synthetic, generated to mirror the structure of production card data (customer
history, fraud bursts, chargeback delay, drift). Absolute metrics will differ on real portfolios;
the pipeline, evaluation design and decisions carry over.
