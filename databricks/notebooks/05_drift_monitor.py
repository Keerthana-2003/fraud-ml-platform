# Databricks notebook source
# MAGIC %run ./_setup

# COMMAND ----------
from fraudml.spark_jobs import drift_check

result = drift_check(spark, T("silver_features"), T("monitoring_drift"), (0, 60), (75, 90))  # noqa: F821
print(result)
dbutils.jobs.taskValues.set("retrain", result["retrain"])  # noqa: F821  (downstream tasks can branch on it)
