"""Reinforcement learning for the review queue: a contextual multi-armed bandit.

Each day, for each customer segment (the context), the policy picks an alert rate (the action):
how much of that segment's traffic analysts review. Reward = fraud dollars caught - review cost.
Thompson sampling learns which alert rate pays off per segment while still exploring, instead of
one global threshold set by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALERT_RATES = (0.002, 0.005, 0.01, 0.02, 0.05)


def daily_rewards(frame: pd.DataFrame, rates=ALERT_RATES, review_cost: float = 15.0) -> dict:
    """Pre-compute reward[(segment, day)][action] from model scores. frame needs
    columns: day, segment, score, is_fraud, amount."""
    table = {}
    for (seg, day), g in frame.groupby(["segment", "day"]):
        order = g.sort_values("score", ascending=False)
        caught = (order["amount"] * order["is_fraud"]).cumsum().to_numpy()
        rewards = []
        for r in rates:
            k = max(1, int(round(len(order) * r)))
            rewards.append(float(caught[k - 1] - review_cost * k))
        table[(seg, day)] = np.asarray(rewards)
    return table


class GaussianThompson:
    """Per-context Gaussian Thompson sampling with a normal prior on each arm's mean reward."""

    def __init__(self, n_arms: int, prior_mean: float = 0.0, prior_var: float = 1e6, noise_var: float | None = None,
                 seed: int = 0):
        self.n_arms, self.prior_mean, self.prior_var, self.noise_var = n_arms, prior_mean, prior_var, noise_var
        self.rng = np.random.default_rng(seed)
        self.stats: dict = {}

    def _s(self, ctx):
        return self.stats.setdefault(ctx, {"n": np.zeros(self.n_arms), "sum": np.zeros(self.n_arms),
                                           "sq": np.zeros(self.n_arms)})

    def choose(self, ctx) -> int:
        s = self._s(ctx)
        n = s["n"]
        mean = np.where(n > 0, s["sum"] / np.maximum(n, 1), 0.0)
        var_obs = np.where(n > 1, (s["sq"] - n * mean ** 2) / np.maximum(n - 1, 1), 0.0)
        noise = self.noise_var if self.noise_var is not None else max(float(np.nanmean(var_obs[n > 1])) if (n > 1).any() else 1e4, 1.0)
        post_var = 1 / (1 / self.prior_var + n / noise)
        post_mean = post_var * (self.prior_mean / self.prior_var + s["sum"] / noise)
        return int(np.argmax(self.rng.normal(post_mean, np.sqrt(post_var))))

    def update(self, ctx, arm: int, reward: float) -> None:
        s = self._s(ctx)
        s["n"][arm] += 1
        s["sum"][arm] += reward
        s["sq"][arm] += reward ** 2


def simulate(table: dict, rounds: int = 400, seed: int = 0, fixed_arm: int = 2) -> dict:
    """Replay days in random order. Compares the bandit with a fixed global alert rate and with the
    best fixed rate per segment chosen in hindsight (oracle)."""
    rng = np.random.default_rng(seed)
    keys = list(table)
    segments = sorted({k[0] for k in keys})
    oracle_arm = {s: int(np.argmax(np.mean([table[k] for k in keys if k[0] == s], axis=0))) for s in segments}
    agent = GaussianThompson(len(next(iter(table.values()))), seed=seed)
    bandit = fixed = oracle = 0.0
    curve = []
    for t in range(rounds):
        seg, day = keys[rng.integers(len(keys))]
        r = table[(seg, day)]
        a = agent.choose(seg)
        agent.update(seg, a, r[a])
        bandit += r[a]
        fixed += r[fixed_arm]
        oracle += r[oracle_arm[seg]]
        if (t + 1) % max(rounds // 10, 1) == 0:
            curve.append({"round": t + 1, "bandit": round(bandit), "fixed": round(fixed), "oracle": round(oracle)})
    learned = {s: int(np.argmax(agent._s(s)["n"])) for s in segments}
    return {"rounds": rounds, "bandit_reward": round(bandit), "fixed_rate_reward": round(fixed),
            "oracle_reward": round(oracle), "regret_vs_oracle": round(oracle - bandit),
            "uplift_vs_fixed_pct": round(100 * (bandit - fixed) / abs(fixed), 1) if fixed else None,
            "learned_alert_rate": {s: ALERT_RATES[a] for s, a in learned.items()},
            "oracle_alert_rate": {s: ALERT_RATES[a] for s, a in oracle_arm.items()}, "curve": curve}
