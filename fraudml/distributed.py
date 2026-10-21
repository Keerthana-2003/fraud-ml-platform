"""Distributed feature preparation.

Customers are independent for point-in-time features, so the data is sharded by customer and each
shard is processed in parallel: on a Ray cluster when Ray is available (local or remote via
RAY_ADDRESS), otherwise in a local process pool. Both paths return identical results; the same
sharding works for Spark (`groupBy(customer_id).applyInPandas`).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .features import build_features


def shard_by_customer(df: pd.DataFrame, n_shards: int) -> list[pd.DataFrame]:
    ids = df["customer_id"].unique()
    buckets = np.arange(len(ids)) % n_shards
    lookup = pd.Series(buckets, index=ids)
    shard_of = lookup.loc[df["customer_id"]].to_numpy()
    return [df[shard_of == s] for s in range(n_shards) if (shard_of == s).any()]


def _ray_available() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def build_features_distributed(df: pd.DataFrame, n_shards: int | None = None, backend: str = "auto") -> tuple[pd.DataFrame, dict]:
    """Returns (features aligned to df.index, run info with backend, shards and wall time)."""
    n_shards = n_shards or max(2, (os.cpu_count() or 2) * 2)
    shards = shard_by_customer(df, n_shards)
    if backend == "auto":
        backend = "ray" if _ray_available() else "processes"
    t0 = time.perf_counter()
    if backend == "ray":
        import ray

        started_here = not ray.is_initialized()
        if started_here:
            ray.init(address=os.getenv("RAY_ADDRESS"), ignore_reinit_error=True, include_dashboard=False,
                     log_to_driver=False)
        remote = ray.remote(build_features)
        parts = ray.get([remote.remote(s) for s in shards])
        if started_here:
            ray.shutdown()
    elif backend == "processes":
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            parts = list(pool.map(build_features, shards))
    else:
        parts = [build_features(s) for s in shards]
    feats = pd.concat(parts).loc[df.index]
    return feats, {"backend": backend, "shards": len(shards), "rows": len(df),
                   "seconds": round(time.perf_counter() - t0, 2)}
