"""Synthetic card-transaction generator with realistic fraud patterns, label delay and drift.

Real fraud datasets are either anonymised (no customer history, so no behavioural features) or
private, so the platform ships with a generator whose structure mirrors production data:

* customers belong to latent behaviour segments (students, families, business travellers, retirees)
  with their own spend levels, active hours, home locations and favourite merchants
* fraud arrives as bursts after a card is compromised: card testing (many tiny online charges,
  then a large one) and account takeover (large charges at new merchants, far from home, odd hours)
* fraud labels arrive late (chargebacks take days to weeks), which motivates semi-supervised learning
* from `drift_day` onward, behaviour shifts (more online spend, price inflation, more card testing),
  which motivates drift monitoring and retraining
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "restaurant", "fuel", "travel", "electronics", "gift_cards", "online_retail", "pharmacy"]
SEGMENTS = {
    #               log-amount mean, active hour, txns/day, online share, category weights
    "student":      (2.6, 19.0, 1.4, 0.55, [0.20, 0.30, 0.05, 0.02, 0.08, 0.02, 0.30, 0.03]),
    "family":       (3.6, 13.0, 1.1, 0.30, [0.40, 0.15, 0.15, 0.02, 0.05, 0.01, 0.17, 0.05]),
    "business":     (4.3, 10.0, 0.9, 0.35, [0.05, 0.30, 0.15, 0.30, 0.08, 0.01, 0.10, 0.01]),
    "retiree":      (3.2, 11.0, 0.6, 0.10, [0.40, 0.15, 0.10, 0.05, 0.02, 0.01, 0.07, 0.20]),
}


@dataclass
class GeneratorConfig:
    n_customers: int = 2000
    days: int = 90
    n_merchants: int = 600
    compromise_rate: float = 0.10
    drift_day: int = 60
    seed: int = 7


def _haversine_offset(lat, lon, km, bearing):
    dlat = km / 111.0 * np.cos(bearing)
    dlon = km / (111.0 * np.cos(np.radians(lat))) * np.sin(bearing)
    return lat + dlat, lon + dlon


def generate(cfg: GeneratorConfig = GeneratorConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    merchant_cat = rng.choice(len(CATEGORIES), cfg.n_merchants)
    merchants_by_cat = [np.where(merchant_cat == c)[0] for c in range(len(CATEGORIES))]
    seg_names = list(SEGMENTS)
    rows = []
    horizon = cfg.days * 86400.0
    drift_t = cfg.drift_day * 86400.0

    for cid in range(cfg.n_customers):
        seg = rng.choice(seg_names, p=[0.25, 0.35, 0.15, 0.25])
        mu, hour_c, rate, online, cat_w = SEGMENTS[seg]
        mu += rng.normal(0, 0.3)
        sigma = rng.uniform(0.4, 0.8)
        hour_c = (hour_c + rng.normal(0, 1.5)) % 24
        home = (rng.uniform(25, 48), rng.uniform(-122, -71))
        cat_w = np.asarray(cat_w) * rng.dirichlet(np.full(len(CATEGORIES), 8.0)) * len(CATEGORIES)
        cat_w /= cat_w.sum()
        favourites = [rng.choice(merchants_by_cat[c], size=min(3, len(merchants_by_cat[c])), replace=False)
                      for c in range(len(CATEGORIES))]

        # legitimate transactions
        start = max(0.0, rng.uniform(-40, cfg.days - 10))  # customers keep joining throughout the period
        n = rng.poisson(rate * (cfg.days - start) * rng.uniform(0.6, 1.4))
        day = np.sort(rng.uniform(start, cfg.days, n)).astype(int)
        hour = np.where(rng.random(n) < 0.1, rng.uniform(0, 24, n), rng.normal(hour_c, 2.5, n)) % 24
        ts = day * 86400.0 + hour * 3600.0
        # legitimate shopping sprees: several purchases within a couple of hours
        spree = rng.random(n) < 0.04
        extra = []
        for i in np.where(spree)[0]:
            extra += list(ts[i] + np.cumsum(rng.uniform(300, 2400, rng.integers(2, 5))))
        ts = np.sort(np.concatenate([ts, np.minimum(extra, horizon - 1)])) if extra else ts
        n = len(ts)
        after = ts >= drift_t
        online_p = np.where(after, min(online + 0.2, 0.9), online)
        is_online = rng.random(n) < online_p
        cats = rng.choice(len(CATEGORIES), n, p=cat_w)
        cats = np.where(is_online & (rng.random(n) < 0.5), CATEGORIES.index("online_retail"), cats)
        amount = np.exp(rng.normal(mu, sigma, n)) * np.where(after, 1.15, 1.0)  # price inflation after drift
        big = rng.random(n) < 0.03  # legitimate big-ticket purchases (laptops, flights, gift cards)
        amount = np.where(big, amount * rng.uniform(3, 12, n), amount)
        cats = np.where(big, rng.choice([CATEGORIES.index(x) for x in ("electronics", "travel", "gift_cards")], n), cats)
        merch = np.array([rng.choice(favourites[c]) if rng.random() < 0.65 else rng.choice(merchants_by_cat[c])
                          for c in cats], dtype=int)
        lat, lon = _haversine_offset(home[0], home[1], np.where(is_online, 0, rng.exponential(6, n)),
                                     rng.uniform(0, 2 * np.pi, n))
        travel = (seg == "business") & (rng.random(n) < 0.08)  # legitimate trips far from home
        lat2, lon2 = _haversine_offset(home[0], home[1], rng.uniform(300, 2500, n), rng.uniform(0, 2 * np.pi, n))
        lat, lon = np.where(travel, lat2, lat), np.where(travel, lon2, lon)
        for i in range(n):
            rows.append((cid, seg, ts[i], amount[i], merch[i], CATEGORIES[cats[i]], lat[i], lon[i],
                         bool(is_online[i]), 0, "none", np.nan))

        # fraud burst after compromise
        if rng.random() < cfg.compromise_rate:
            t0 = rng.uniform((start + 5) * 86400, horizon - 86400) if start + 5 < cfg.days - 1 else horizon - 2 * 86400
            testing_p = 0.6 if t0 >= drift_t else 0.35
            delay = rng.uniform(2, 40) * 86400  # chargeback arrives later
            if rng.random() < testing_p:  # card testing
                k = rng.integers(2, 7)
                t = t0 + np.cumsum(rng.uniform(60, 3600, k))
                amts = list(rng.uniform(1, 15, k - 1)) + [np.exp(mu) * rng.uniform(2, 8)]
                kind = "card_testing"
            else:  # account takeover
                k = rng.integers(1, 5)
                t = t0 + np.cumsum(rng.uniform(600, 14400, k))
                amts = np.exp(mu) * rng.uniform(1.5, 8, k)
                kind = "account_takeover"
            local = rng.random() < 0.25  # skimmed card used close to home
            far_lat, far_lon = _haversine_offset(home[0], home[1], rng.uniform(5, 30) if local else rng.uniform(800, 4000),
                                                 rng.uniform(0, 2 * np.pi))
            if rng.random() < 0.35:  # fraudster active during the customer's normal hours
                t = np.floor(t / 86400) * 86400 + ((hour_c + rng.normal(0, 2, k)) % 24) * 3600 + np.arange(k) * 900
            for j in range(k):
                pool = ("electronics", "gift_cards", "online_retail") if rng.random() < 0.7 else ("grocery", "fuel", "restaurant")
                c = rng.choice([CATEGORIES.index(x) for x in pool])
                online_f = kind == "card_testing" or rng.random() < 0.5
                rows.append((cid, seg, min(t[j], horizon - 1), float(amts[j]), int(rng.choice(merchants_by_cat[c])),
                             CATEGORIES[c], far_lat if not online_f else home[0], far_lon if not online_f else home[1],
                             bool(online_f), 1, kind, t[j] + delay))

    df = pd.DataFrame(rows, columns=["customer_id", "segment", "ts", "amount", "merchant_id", "category", "lat", "lon",
                                     "is_online", "is_fraud", "fraud_type", "label_ts"])
    df = df.sort_values(["ts", "customer_id"], kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(len(df)))
    df["day"] = (df["ts"] // 86400).astype(int)
    df["amount"] = df["amount"].round(2)
    return df
