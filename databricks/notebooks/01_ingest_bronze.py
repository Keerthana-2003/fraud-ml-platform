# Databricks notebook source
# MAGIC %run ./_setup

# COMMAND ----------
from fraudml.data import GeneratorConfig, generate
from fraudml.spark_jobs import write_bronze

# Replace generate() with your source (Auto Loader, Kafka, JDBC) in a real deployment.
transactions = generate(GeneratorConfig())
rows = write_bronze(spark, transactions, T("bronze_transactions"))  # noqa: F821
print(f"bronze rows: {rows:,}")
