# Databricks notebook source
# MAGIC %run ./_setup

# COMMAND ----------
import mlflow

from fraudml.spark_jobs import batch_score

mlflow.set_registry_uri("databricks-uc")
rows = batch_score(spark, T("silver_features"), T("gold_scores"), MODEL_NAME, from_day=75)  # noqa: F821
print(f"scored rows: {rows:,}")
