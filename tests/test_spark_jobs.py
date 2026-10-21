"""Databricks pipeline on a local SparkSession with Delta Lake and a local MLflow registry (runs in CI)."""
from __future__ import annotations

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
pytest.importorskip("delta")
mlflow = pytest.importorskip("mlflow")

from fraudml.data import GeneratorConfig, generate  # noqa: E402
from fraudml.features import FEATURES, build_features  # noqa: E402
from fraudml.spark_jobs import batch_score, build_silver, drift_check, train_and_register, write_bronze  # noqa: E402


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    wh = tmp_path_factory.mktemp("warehouse")
    builder = (SparkSession.builder.master("local[2]").appName("fraudml-test")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.sql.warehouse.dir", str(wh))
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
    session = configure_spark_with_delta_pip(builder).getOrCreate()
    session.sql("CREATE DATABASE IF NOT EXISTS fraud")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path_factory.mktemp('mlflow')}/mlflow.db")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def data():
    return generate(GeneratorConfig(n_customers=150, days=40, drift_day=25, seed=3))


def test_medallion_pipeline_end_to_end(spark, data):
    assert write_bronze(spark, data, "fraud.bronze_transactions") == len(data)
    assert build_silver(spark, "fraud.bronze_transactions", "fraud.silver_features") == len(data)

    # distributed features must equal the single-machine features exactly
    silver = spark.read.table("fraud.silver_features").toPandas().set_index("txn_id").sort_index()
    local = build_features(data).set_index(data["txn_id"]).sort_index()
    np.testing.assert_allclose(silver[FEATURES].to_numpy(), local[FEATURES].to_numpy(), rtol=1e-9, atol=1e-9)

    first = train_and_register(spark, "fraud.silver_features", "fraud_model", "/fraud-test", 25, 32)
    assert first["promoted_to_champion"] and first["version"] == 1 and 0 <= first["metrics"]["pr_auc"] <= 1
    second = train_and_register(spark, "fraud.silver_features", "fraud_model", "/fraud-test", 25, 32)
    assert second["version"] == 2 and not second["promoted_to_champion"]  # identical model: no 0.01 gain

    n = batch_score(spark, "fraud.silver_features", "fraud.gold_scores", "fraud_model", from_day=32)
    gold = spark.read.table("fraud.gold_scores").toPandas()
    assert n == int((data["day"] >= 32).sum()) and gold["fraud_score"].between(0, 1).all()
    assert gold["score_percentile"].between(0, 1).all()

    d = drift_check(spark, "fraud.silver_features", "fraud.monitoring_drift", (0, 25), (25, 40))
    assert set(d) == {"max_psi", "worst_feature", "features_over_threshold", "retrain"}
    assert spark.read.table("fraud.monitoring_drift").count() == len(FEATURES)
