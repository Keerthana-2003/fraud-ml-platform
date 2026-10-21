"""Point-in-time behavioural features: every feature for a transaction uses only that customer's
earlier transactions, so training data never leaks the future (the most common bug in fraud models).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORIES

FEATURES = [
    "log_amount", "amount_z", "amount_vs_median", "log_secs_since_last", "count_1h", "count_24h",
    "log_sum_24h", "small_count_1h", "is_new_merchant", "is_new_category", "log_dist_home_km",
    "hour_sin", "hour_cos", "hour_deviation", "is_online", "history_depth",
] + [f"cat_{c}" for c in CATEGORIES]


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def customer_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features for one customer's transactions (any order in, original index preserved)."""
    g = g.sort_values("ts", kind="stable")
    ts, amt = g["ts"].to_numpy(), g["amount"].to_numpy()
    la, lo = g["lat"].to_numpy(), g["lon"].to_numpy()
    merch, cat = g["merchant_id"].to_numpy(), g["category"].to_numpy()
    online = g["is_online"].to_numpy()
    n = len(g)
    out = {k: np.zeros(n) for k in FEATURES}
    log_amt = np.log1p(amt)
    hours = (ts % 86400) / 3600
    seen_m, seen_c = set(), set()
    lo_1h = lo_24h = 0
    csum_amt = np.concatenate([[0.0], np.cumsum(amt)])
    small = np.concatenate([[0], np.cumsum(amt < 10)])
    for i in range(n):
        while ts[lo_1h] < ts[i] - 3600:
            lo_1h += 1
        while ts[lo_24h] < ts[i] - 86400:
            lo_24h += 1
        hist = slice(0, i)
        k = i
        out["count_1h"][i] = i - lo_1h
        out["count_24h"][i] = i - lo_24h
        out["log_sum_24h"][i] = np.log1p(csum_amt[i] - csum_amt[lo_24h])
        out["small_count_1h"][i] = small[i] - small[lo_1h]
        out["history_depth"][i] = min(k, 20) / 20  # saturates, so it does not drift with calendar time
        if k >= 3:
            prev = log_amt[hist]
            out["amount_z"][i] = (log_amt[i] - prev.mean()) / (prev.std() + 0.25)
            out["amount_vs_median"][i] = amt[i] / (np.median(amt[hist]) + 1)
            home_lat, home_lon = np.median(la[hist][~online[hist]]) if (~online[hist]).any() else la[i], \
                np.median(lo[hist][~online[hist]]) if (~online[hist]).any() else lo[i]
            out["log_dist_home_km"][i] = 0.0 if online[i] else np.log1p(_haversine(home_lat, home_lon, la[i], lo[i]))
            ang = hours[hist] / 24 * 2 * np.pi
            mean_h = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * 24) % 24
            d = abs(hours[i] - mean_h)
            out["hour_deviation"][i] = min(d, 24 - d)
        else:
            out["amount_vs_median"][i] = 1.0
        out["log_secs_since_last"][i] = np.log1p(ts[i] - ts[i - 1]) if i else np.log1p(86400 * 30)
        out["is_new_merchant"][i] = float(merch[i] not in seen_m)
        out["is_new_category"][i] = float(cat[i] not in seen_c)
        seen_m.add(merch[i])
        seen_c.add(cat[i])
    out["log_amount"] = log_amt
    out["hour_sin"], out["hour_cos"] = np.sin(hours / 24 * 2 * np.pi), np.cos(hours / 24 * 2 * np.pi)
    out["is_online"] = online.astype(float)
    for c in CATEGORIES:
        out[f"cat_{c}"] = (cat == c).astype(float)
    return pd.DataFrame(out, index=g.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [customer_features(g) for _, g in df.groupby("customer_id", sort=False)]
    return pd.concat(parts).loc[df.index]


def customer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer behaviour vectors (spend level, category mix, hour histogram, online share)
    used for clustering and PCA. Built from legitimate history only."""
    legit = df[df["is_fraud"] == 0]
    hours = ((legit["ts"] % 86400) // 3600).astype(int)
    cat_mix = pd.crosstab(legit["customer_id"], legit["category"], normalize="index").reindex(columns=CATEGORIES, fill_value=0)
    hour_hist = pd.crosstab(legit["customer_id"], hours // 3, normalize="index").reindex(columns=range(8), fill_value=0)
    hour_hist.columns = [f"h{3 * h:02d}" for h in hour_hist.columns]
    g = legit.groupby("customer_id")
    days = max(df["day"].max() - df["day"].min() + 1, 1)
    base = pd.DataFrame({"log_amount_mean": np.log1p(g["amount"].median()), "txns_per_day": g.size() / days,
                         "online_share": g["is_online"].mean()})
    return base.join(cat_mix.add_prefix("mix_")).join(hour_hist).fillna(0)
