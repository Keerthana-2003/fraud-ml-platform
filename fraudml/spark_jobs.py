"""Databricks / Spark implementation of the pipeline (medallion architecture on Delta Lake).

  bronze  raw transactions (append-only)
  silver  point-in-time features, computed per customer with groupBy().applyInPandas, so the same
          pandas feature code runs distributed across the cluster
  gold    model scores for the latest window, ready for the review queue and dashboards

Training logs parameters, metrics and the model to MLflow and registers it in the model registry
(Unity Catalog on Databricks) with a `champion` alias; batch scoring loads `models:/<name>@champion`
as a Spark UDF. The same functions run on Databricks (notebooks in databricks/notebooks) and on a
local SparkSession with Delta Lake (tests/test_spark_jobs.py in CI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, customer_features
from .modeling import fraud_metrics, gradient_boosting
from .monitoring import psi

BRONZE_COLUMNS = ["txn_id", "customer_id", "segment", "ts", "amount", "merchant_id", "category", "lat", "lon",
                  "is_online", "is_fraud", "fraud_type", "label_ts", "day"]


class FraudScorer:
    """MLflow pyfunc wrapper returning the fraud probability (not the 0/1 class) so batch scoring,
    model serving and the Spark UDF all get scores."""

    def __new__(cls, model):
        import mlflow

        class _Scorer(mlflow.pyfunc.PythonModel):
            def __init__(self, m):
                self.model = m

            def predict(self, context, model_input, params=None):
                return self.model.predict_proba(model_input[FEATURES])[:, 1]

        return _Scorer(model)


def _log_pyfunc(mlflow, python_model, signature, registered_model_name):
    from pathlib import Path

    code = [str(Path(__file__).resolve().parent)]  # ship the fraudml package with the model
    kwargs = dict(python_model=python_model, signature=signature, registered_model_name=registered_model_name,
                  code_paths=code)
    try:
        return mlflow.pyfunc.log_model(name="model", **kwargs)          # MLflow 3
    except TypeError:
        return mlflow.pyfunc.log_model(artifact_path="model", **kwargs)  # MLflow 2


def write_bronze(spark, transactions: pd.DataFrame, table: str, mode: str = "overwrite") -> int:
    sdf = spark.createDataFrame(transactions[BRONZE_COLUMNS])
    sdf.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable(table)
    return sdf.count()


def _silver_schema() -> str:
    feats = ", ".join(f"{f} double" for f in FEATURES)
    return f"txn_id long, customer_id long, day long, segment string, amount double, is_fraud long, {feats}"


def _features_for_customer(pdf: pd.DataFrame) -> pd.DataFrame:
    feats = customer_features(pdf.set_index("txn_id", drop=False))
    out = pdf.set_index("txn_id", drop=False)[["txn_id", "customer_id", "day", "segment", "amount", "is_fraud"]]
    return out.join(feats).reset_index(drop=True)


def build_silver(spark, bronze_table: str, silver_table: str) -> int:
    """Point-in-time features, distributed by customer across the Spark cluster."""
    bronze = spark.read.table(bronze_table)
    silver = bronze.groupBy("customer_id").applyInPandas(_features_for_customer, schema=_silver_schema())
    silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_table)
    return spark.read.table(silver_table).count()


def train_and_register(spark, silver_table: str, model_name: str, experiment: str, train_until_day: int,
                       valid_until_day: int, params: dict | None = None) -> dict:
    """Train on days < train_until_day, validate on the next window, log to MLflow, register the
    model and point the `champion` alias at it if it beats the current champion."""
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    pdf = spark.read.table(silver_table).where(f"day < {valid_until_day}").toPandas()
    tr, va = pdf["day"] < train_until_day, pdf["day"] >= train_until_day
    X, y = pdf[FEATURES], pdf["is_fraud"].to_numpy()
    params = params or {"learning_rate": 0.08, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 0.5}
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="fraud-gbm") as run:
        model = gradient_boosting(**params).fit(X[tr], y[tr])
        metrics = fraud_metrics(y[va], model.predict_proba(X[va])[:, 1], pdf.loc[va, "amount"].to_numpy())
        mlflow.log_params({**params, "train_until_day": train_until_day, "valid_until_day": valid_until_day,
                           "silver_table": silver_table})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        sample = X[tr].head(100)
        info = _log_pyfunc(mlflow, FraudScorer(model), infer_signature(sample, model.predict_proba(sample)[:, 1]),
                           model_name)
    client = MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(f"name='{model_name}'"))
    try:
        current = client.get_model_version_by_alias(model_name, "champion")
        current_pr = client.get_run(current.run_id).data.metrics.get("pr_auc", -1.0)
    except Exception:  # no champion yet
        current_pr = -1.0
    promoted = metrics["pr_auc"] >= current_pr + 0.01 or current_pr < 0
    if promoted:
        client.set_registered_model_alias(model_name, "champion", str(version))
    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "version": version, "metrics": metrics,
            "promoted_to_champion": promoted, "previous_champion_pr_auc": current_pr}


def batch_score(spark, silver_table: str, gold_table: str, model_name: str, from_day: int) -> int:
    """Score the latest window with the champion model as a Spark UDF and write the gold table."""
    import mlflow
    from pyspark.sql import functions as F

    udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}@champion", result_type="double", env_manager="local")
    scored = (spark.read.table(silver_table).where(F.col("day") >= from_day)
              .withColumn("fraud_score", udf(*[F.col(f) for f in FEATURES]))
              .select("txn_id", "customer_id", "day", "segment", "amount", "fraud_score",
                      F.percent_rank().over(_window_by_day()).alias("score_percentile")))
    scored.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(gold_table)
    return spark.read.table(gold_table).count()


def _window_by_day():
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    return Window.partitionBy("day").orderBy(F.col("fraud_score"))


def drift_check(spark, silver_table: str, monitor_table: str, reference_days: tuple[int, int],
                current_days: tuple[int, int], threshold: float = 0.2, sample: int = 50000) -> dict:
    """PSI of every feature between a reference and a current window; appends one row per feature to
    a monitoring Delta table (for dashboards and alerts) and returns the retrain decision."""
    silver = spark.read.table(silver_table)
    ref = silver.where(f"day >= {reference_days[0]} AND day < {reference_days[1]}").limit(sample).toPandas()
    cur = silver.where(f"day >= {current_days[0]} AND day < {current_days[1]}").limit(sample).toPandas()
    rows = [{"feature": f, "psi": float(psi(ref[f].to_numpy(), cur[f].to_numpy())),
             "reference_days": f"{reference_days[0]}-{reference_days[1] - 1}",
             "current_days": f"{current_days[0]}-{current_days[1] - 1}"} for f in FEATURES]
    spark.createDataFrame(pd.DataFrame(rows)).write.format("delta").mode("append").saveAsTable(monitor_table)
    worst = max(rows, key=lambda r: r["psi"])
    return {"max_psi": round(worst["psi"], 4), "worst_feature": worst["feature"],
            "features_over_threshold": int(np.sum([r["psi"] > threshold for r in rows])),
            "retrain": worst["psi"] > threshold}
