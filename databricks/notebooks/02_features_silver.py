# Databricks notebook source
# MAGIC %run ./_setup

# COMMAND ----------
from fraudml.spark_jobs import build_silver

rows = build_silver(spark, T("bronze_transactions"), T("silver_features"))  # noqa: F821
print(f"silver rows: {rows:,}")
