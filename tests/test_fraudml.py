from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudml.bandit import GaussianThompson, simulate
from fraudml.data import GeneratorConfig, generate
from fraudml.distributed import build_features_distributed, shard_by_customer
from fraudml.features import FEATURES, build_features, customer_features
from fraudml.modeling import diagnose, fraud_metrics, gradient_boosting
from fraudml.monitoring import ModelRegistry, RetrainPolicy, drift_report, psi
from fraudml.responsible import local_explanation, slice_metrics
from fraudml.semi_supervised import labels_known_at, reviewed_sample, self_training
from fraudml.sequence import build_sequences
from fraudml.unsupervised import BehaviourSegments


@pytest.fixture(scope="module")
def small():
    df = generate(GeneratorConfig(n_customers=150, days=40, drift_day=25, seed=3))
    return df, build_features(df)


def test_generator_structure(small):
    df, _ = small
    assert df["ts"].is_monotonic_increasing and df["is_fraud"].sum() > 0
    fraud = df[df["is_fraud"] == 1]
    assert (fraud["label_ts"] > fraud["ts"]).all() and df.loc[df["is_fraud"] == 0, "label_ts"].isna().all()
    legit = df[df["is_fraud"] == 0]
    assert legit.loc[legit["day"] >= 25, "is_online"].mean() > legit.loc[legit["day"] < 25, "is_online"].mean()


def test_features_are_point_in_time(small):
    """Removing a customer's future transactions must not change features of earlier ones."""
    df, feats = small
    cust = df["customer_id"].value_counts().index[0]
    g = df[df["customer_id"] == cust]
    cut = g.iloc[: len(g) // 2]
    np.testing.assert_allclose(customer_features(cut).to_numpy(), feats.loc[cut.index].to_numpy())
    assert list(feats.columns) == FEATURES


def test_distributed_matches_single_process(small):
    df, feats = small
    out, info = build_features_distributed(df, n_shards=3, backend="processes")
    pd.testing.assert_frame_equal(out, feats)
    assert info["shards"] == 3 and sum(len(s) for s in shard_by_customer(df, 3)) == len(df)


def test_distributed_ray_backend(small):
    pytest.importorskip("ray")
    df, feats = small
    out, info = build_features_distributed(df, n_shards=2, backend="ray")
    assert info["backend"] == "ray"
    pd.testing.assert_frame_equal(out, feats)


def test_fraud_metrics_and_budget():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    m = fraud_metrics(y, np.linspace(1, 0, 10), amount=np.full(10, 100.0), alert_rate=0.2)
    assert m["precision_at_20.0%"] == 0.5 and m["recall_at_20.0%"] == 0.5 and m["dollar_recall_at_20.0%"] == 0.5
    assert diagnose(0.3, 0.29).startswith("underfitting") and diagnose(1.0, 0.5).startswith("overfitting")
    assert diagnose(0.8, 0.75) == "good fit"


def test_gradient_boosting_learns_signal(small):
    df, feats = small
    tr = df["day"] < 30
    model = gradient_boosting(max_iter=100).fit(feats[tr], df["is_fraud"][tr])
    s = model.predict_proba(feats[~tr])[:, 1]
    assert fraud_metrics(df["is_fraud"][~tr], s)["roc_auc"] > 0.8


def test_segments_recover_clusters():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, (3, 6))
    X = np.vstack([c + rng.normal(0, 0.5, (100, 6)) for c in centers])
    truth = pd.Series(np.repeat([0, 1, 2], 100))
    prof = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    seg = BehaviourSegments(n_components=3).fit(prof)
    assert seg.k == 3 and seg.agreement(prof, truth) > 0.95


def test_label_delay_and_self_training(small):
    df, feats = small
    known = labels_known_at(df, df["ts"].quantile(0.5))
    assert known.sum() < df["is_fraud"].sum()
    lab = reviewed_sample(len(df), known, 0.2)
    assert lab[known == 1].all()
    model = self_training(max_iter=2).fit(feats, np.where(lab, known, -1))
    assert model.predict_proba(feats).shape == (len(df), 2)


def test_bandit_finds_best_arm():
    rng = np.random.default_rng(0)
    agent = GaussianThompson(3, seed=0)
    means = [1.0, 5.0, 2.0]
    for _ in range(300):
        a = agent.choose("ctx")
        agent.update("ctx", a, rng.normal(means[a], 1.0))
    assert np.argmax(agent._s("ctx")["n"]) == 1
    table = {(s, d): np.array([0.0, 10.0 if s == "a" else 0.0, 5.0 if s == "b" else 0.0]) + rng.normal(0, 0.5, 3)
             for s in "ab" for d in range(20)}
    res = simulate(table, rounds=400, fixed_arm=0)
    assert res["learned_alert_rate"]["a"] != res["learned_alert_rate"]["b"] and res["bandit_reward"] > res["fixed_rate_reward"]


def test_psi_and_retrain_policy():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02 and psi(ref, rng.normal(1, 1, 5000)) > 0.2
    assert psi(np.r_[np.zeros(900), np.ones(100)], np.r_[np.zeros(500), np.ones(500)]) > 0.2  # binary feature
    rep = drift_report(pd.DataFrame({"a": ref, "b": ref}), pd.DataFrame({"a": ref, "b": ref + 2}))
    assert list(rep["top_features"])[0] == "b" and rep["features_over_0.2"] == 1
    p = RetrainPolicy()
    assert p.should_retrain(0.3, 0.0, None)[0] and not p.should_retrain(0.1, 0.05, 0.02)[0]
    assert p.promote(0.40, 0.42) and not p.promote(0.40, 0.405)


def test_registry_promotion(tmp_path):
    reg = ModelRegistry(tmp_path)
    a = reg.register({"m": 1}, ["x"], "w1", {"pr_auc": 0.4}, stage="champion")
    b = reg.register({"m": 2}, ["x"], "w2", {"pr_auc": 0.5})
    assert reg.champion()["version"] == a.version
    reg.promote(b.version, "better")
    assert reg.champion()["version"] == b.version and reg.load(b.version) == {"m": 2}
    assert ModelRegistry(tmp_path).champion()["version"] == b.version  # persisted


def test_slices_and_explanations(small):
    df, feats = small
    y = df["is_fraud"].to_numpy()
    model = gradient_boosting(max_iter=50).fit(feats, y)
    s = model.predict_proba(feats)[:, 1]
    res = slice_metrics(y, s, df["segment"], threshold=np.quantile(s, 0.99))
    assert set(res["by_group"]) == set(df["segment"].unique())
    exp = local_explanation(model, feats.reset_index(drop=True), int(np.argmax(s)), top=3)
    assert len(exp["top_contributions"]) == 3 and exp["score"] > 0.5


def test_sequences_are_ordered_and_padded(small):
    df, feats = small
    seqs = build_sequences(df, feats, window=4)
    cust = df["customer_id"].iloc[0]
    rows = np.where(df["customer_id"] == cust)[0]
    assert seqs.shape == (len(df), 4, feats.shape[1])
    np.testing.assert_allclose(seqs[rows[0], -1], feats.iloc[rows[0]].to_numpy())
    assert np.all(seqs[rows[0], :-1] == 0)  # first transaction: no history, left padding
    np.testing.assert_allclose(seqs[rows[2], -2], feats.iloc[rows[1]].to_numpy())


def test_lstm_trains(small):
    pytest.importorskip("torch")
    from fraudml.sequence import LSTMScorer

    df, feats = small
    seqs = build_sequences(df, feats, window=5)
    model = LSTMScorer(feats.shape[1], hidden=16).fit(seqs, df["is_fraud"].to_numpy(), epochs=3, batch=256)
    assert model.history[-1]["train_loss"] < model.history[0]["train_loss"]
    assert model.predict_proba(seqs[:10]).shape == (10,)
