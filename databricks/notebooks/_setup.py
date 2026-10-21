# Databricks notebook source
# Shared setup: make the bundle's fraudml package importable and read job parameters.
import os
import sys

bundle_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if bundle_root not in sys.path:
    sys.path.insert(0, bundle_root)

dbutils.widgets.text("catalog", "main")  # noqa: F821
dbutils.widgets.text("schema", "fraud")  # noqa: F821
CATALOG = dbutils.widgets.get("catalog")  # noqa: F821
SCHEMA = dbutils.widgets.get("schema")  # noqa: F821
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")  # noqa: F821
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"  # noqa: E731
MODEL_NAME = T("fraud_model")  # Unity Catalog registered model
