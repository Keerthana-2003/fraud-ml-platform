# Databricks notebook source
# MAGIC %run ./_setup

# COMMAND ----------
import json

import mlflow

from fraudml.spark_jobs import train_and_register

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]  # noqa: F821
result = train_and_register(spark, T("silver_features"), MODEL_NAME, f"/Users/{user}/fraud-ml-platform",  # noqa: F821
                            train_until_day=60, valid_until_day=75)
print(json.dumps(result, indent=2, default=str))
