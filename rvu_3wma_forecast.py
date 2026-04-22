# Databricks notebook source

# MAGIC %md
# MAGIC # Hourly RVU Prediction — 3-Week Moving Average (3WMA)
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook generates **hourly RVU predictions** for a radiology dataset using a
# MAGIC **3-Week Moving Average (3WMA)** method.
# MAGIC
# MAGIC ### Forecasting Formula
# MAGIC
# MAGIC For each timestamp **t** on the prediction date, the predicted RVU is:
# MAGIC
# MAGIC ```
# MAGIC RVU_predicted(t) = mean( RVU(t − 7 days), RVU(t − 14 days), RVU(t − 21 days) )
# MAGIC ```
# MAGIC
# MAGIC - Matches are **exact** — same hour, same site, same priority.
# MAGIC - If only some lags are available, the mean is taken over available values only.
# MAGIC - If **no** lags are available, prediction is `NULL`.
# MAGIC
# MAGIC ### Example
# MAGIC
# MAGIC | Prediction date | Training window          | Lags used      |
# MAGIC |-----------------|--------------------------|----------------|
# MAGIC | 2026-01-01      | 2025-12-11 → 2025-12-31  | t−7, t−14, t−21 |
# MAGIC
# MAGIC ### Notebook Structure
# MAGIC 1. Widgets / Parameters
# MAGIC 2. Load data (Spark SQL)
# MAGIC 3. Data validation & logging
# MAGIC 4. Forecast computation (3WMA)
# MAGIC 5. Coverage summary
# MAGIC 6. Display results
# MAGIC 7. (Optional) Visualisation — actual vs predicted

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets / Parameters

# COMMAND ----------

dbutils.widgets.removeAll()

dbutils.widgets.text(
    "prediction_date",
    "2026-01-01",
    "Prediction Date (YYYY-MM-DD)",
)

dbutils.widgets.text(
    "catalog_table",
    "edw_dev.matrix_lateetud.exam_data_lateetud_deduped",
    "Catalog Table",
)

# COMMAND ----------

from datetime import datetime, timedelta

PREDICTION_DATE_STR: str = dbutils.widgets.get("prediction_date").strip()
CATALOG_TABLE: str = dbutils.widgets.get("catalog_table").strip()

# Validate date format
try:
    PREDICTION_DATE = datetime.strptime(PREDICTION_DATE_STR, "%Y-%m-%d").date()
except ValueError as exc:
    raise ValueError(
        f"prediction_date '{PREDICTION_DATE_STR}' is not in YYYY-MM-DD format."
    ) from exc

HISTORY_START = PREDICTION_DATE - timedelta(days=21)

SITE_IDS = (1321, 1319, 1344, 1648, 1335, 1320, 1359, 1327, 1342, 1345, 1328, 1336, 1318, 1323)
SITE_IDS_SQL = ", ".join(str(s) for s in SITE_IDS)

print(f"Prediction date : {PREDICTION_DATE_STR}")
print(f"Catalog table   : {CATALOG_TABLE}")
print(f"History window  : {HISTORY_START}  →  {PREDICTION_DATE - timedelta(days=1)}")
print(f"Site IDs        : {SITE_IDS_SQL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data (Spark SQL)

# COMMAND ----------

query = f"""
SELECT
    Modified_Clario_Site_ID,
    Priority,
    Modified_Unread_DTS,
    RVU
FROM {CATALOG_TABLE}
WHERE Modified_Clario_Site_ID IN ({SITE_IDS_SQL})
  AND CAST(Modified_Unread_DTS AS DATE) >= '{HISTORY_START}'
  AND CAST(Modified_Unread_DTS AS DATE) <= '{PREDICTION_DATE_STR}'
"""

print("Executing SQL:\n", query)
raw_spark_df = spark.sql(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Validation & Logging

# COMMAND ----------

# ── Column presence check ───────────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "Modified_Clario_Site_ID",
    "Priority",
    "Modified_Unread_DTS",
    "RVU",
}

missing_cols = REQUIRED_COLUMNS - set(raw_spark_df.columns)
if missing_cols:
    raise ValueError(f"Required columns missing from table: {missing_cols}")

print("✓ All required columns present.")

# ── Row counts ──────────────────────────────────────────────────────────────
total_rows = raw_spark_df.count()
if total_rows == 0:
    raise Exception(
        f"No data returned from {CATALOG_TABLE} for the period "
        f"{HISTORY_START} → {PREDICTION_DATE_STR}. "
        "Check that the table exists and the date range is correct."
    )

print(f"Total rows loaded  : {total_rows:,}")

# ── Timestamp range ─────────────────────────────────────────────────────────
from pyspark.sql import functions as F

ts_stats = raw_spark_df.agg(
    F.min("Modified_Unread_DTS").alias("min_ts"),
    F.max("Modified_Unread_DTS").alias("max_ts"),
).collect()[0]

print(f"Timestamp range    : {ts_stats['min_ts']}  →  {ts_stats['max_ts']}")

# ── Test-set row count ──────────────────────────────────────────────────────
test_spark_df = raw_spark_df.filter(
    F.col("Modified_Unread_DTS").cast("date") == F.lit(PREDICTION_DATE_STR)
)
test_rows = test_spark_df.count()

if test_rows == 0:
    raise Exception(
        f"Test set is empty: no rows found for prediction_date = {PREDICTION_DATE_STR}. "
        "Ensure the table contains data for this date."
    )

print(f"Test-set rows      : {test_rows:,}")

# ── History check ───────────────────────────────────────────────────────────
history_spark_df = raw_spark_df.filter(
    F.col("Modified_Unread_DTS").cast("date") < F.lit(PREDICTION_DATE_STR)
)
history_rows = history_spark_df.count()

if history_rows == 0:
    print(
        "WARNING: No historical rows found in the 21-day window before "
        f"{PREDICTION_DATE_STR}. All predictions will be NULL."
    )
elif history_rows < test_rows:
    print(
        f"WARNING: Historical rows ({history_rows:,}) are fewer than test rows "
        f"({test_rows:,}). Coverage may be low."
    )
else:
    print(f"History rows       : {history_rows:,}  ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Convert to Pandas & Build Lookup Dictionary

# COMMAND ----------

import pandas as pd
import numpy as np

# Convert full dataset (history + test day) to Pandas
full_df: pd.DataFrame = raw_spark_df.toPandas()

# Ensure correct dtypes
full_df["Modified_Clario_Site_ID"] = full_df["Modified_Clario_Site_ID"].astype(int)
full_df["Priority"] = full_df["Priority"].astype(str)
full_df["Modified_Unread_DTS"] = pd.to_datetime(full_df["Modified_Unread_DTS"])
full_df["RVU"] = pd.to_numeric(full_df["RVU"], errors="coerce")

# ── Build lookup: (site_id, priority, timestamp) → RVU ──────────────────────
# For duplicate keys keep the last occurrence (should not happen on a deduped table,
# but guards against unexpected duplicates).
lookup: dict[tuple[int, str, pd.Timestamp], float | None] = (
    full_df
    .set_index(["Modified_Clario_Site_ID", "Priority", "Modified_Unread_DTS"])["RVU"]
    .to_dict()
)

print(f"Lookup table size  : {len(lookup):,} entries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Forecast Computation (3WMA Logic)

# COMMAND ----------

# ── Isolate test set ─────────────────────────────────────────────────────────
test_df: pd.DataFrame = full_df[
    full_df["Modified_Unread_DTS"].dt.date == PREDICTION_DATE
].copy()

# ── Apply 3WMA ───────────────────────────────────────────────────────────────
LAG_DAYS = [7, 14, 21]


def get_lag(site_id: int, priority: str, ts: pd.Timestamp, lag_days: int) -> float | None:
    """Return RVU for exactly (ts - lag_days) or None if not in lookup."""
    lag_ts = ts - pd.Timedelta(days=lag_days)
    return lookup.get((site_id, priority, lag_ts), None)


def compute_3wma(
    site_id: int,
    priority: str,
    ts: pd.Timestamp,
    rvu_actual: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Compute 3WMA prediction.
    Returns (lag_7d, lag_14d, lag_21d, rvu_predicted).
    If rvu_actual is NaN/None the prediction is also NULL.
    """
    if pd.isna(rvu_actual):
        return None, None, None, None

    lag_7d = get_lag(site_id, priority, ts, 7)
    lag_14d = get_lag(site_id, priority, ts, 14)
    lag_21d = get_lag(site_id, priority, ts, 21)

    available = [v for v in (lag_7d, lag_14d, lag_21d) if v is not None and not np.isnan(v)]
    rvu_predicted = float(np.mean(available)) if available else None

    return lag_7d, lag_14d, lag_21d, rvu_predicted


results = test_df.apply(
    lambda row: compute_3wma(
        row["Modified_Clario_Site_ID"],
        row["Priority"],
        row["Modified_Unread_DTS"],
        row["RVU"],
    ),
    axis=1,
    result_type="expand",
)
results.columns = ["lag_7d", "lag_14d", "lag_21d", "RVU_predicted"]

# ── Assemble output dataframe ────────────────────────────────────────────────
predictions_df = test_df[
    ["Modified_Clario_Site_ID", "Priority", "Modified_Unread_DTS", "RVU"]
].rename(columns={"RVU": "RVU_actual"}).copy()

predictions_df = pd.concat([predictions_df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

# Sort as required
predictions_df = predictions_df.sort_values(
    ["Modified_Unread_DTS", "Modified_Clario_Site_ID", "Priority"]
).reset_index(drop=True)

print(f"predictions_df shape: {predictions_df.shape}")
print(predictions_df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Coverage Summary

# COMMAND ----------

def lag_count(row) -> int:
    """Count how many of the three lags are available (non-null) for a row."""
    return sum(
        1
        for v in (row["lag_7d"], row["lag_14d"], row["lag_21d"])
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    )


predictions_df["_lag_count"] = predictions_df.apply(lag_count, axis=1)

coverage = predictions_df["_lag_count"].value_counts().reindex([3, 2, 1, 0], fill_value=0)

print("=" * 45)
print("Coverage Summary")
print("=" * 45)
print(f"  Rows with 3 lags available : {coverage[3]:>6,}")
print(f"  Rows with 2 lags available : {coverage[2]:>6,}")
print(f"  Rows with 1 lag  available : {coverage[1]:>6,}")
print(f"  Rows with 0 lags (NULL)    : {coverage[0]:>6,}")
print("-" * 45)
print(f"  Total test rows            : {len(predictions_df):>6,}")
print("=" * 45)

if coverage[0] == len(predictions_df):
    print(
        "\nWARNING: All predictions are NULL. "
        "No historical data found for any lag window."
    )
elif coverage[0] > 0:
    pct_null = 100.0 * coverage[0] / len(predictions_df)
    print(f"\nNote: {coverage[0]:,} rows ({pct_null:.1f}%) have NULL predictions.")

# Drop helper column
predictions_df.drop(columns=["_lag_count"], inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Display Results

# COMMAND ----------

display(predictions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. (Optional) Visualisation — Actual vs Predicted
# MAGIC
# MAGIC Aggregates RVU by hour across all sites and priorities for a quick sanity check.

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plot_df = (
    predictions_df
    .groupby("Modified_Unread_DTS")[["RVU_actual", "RVU_predicted"]]
    .sum()
    .reset_index()
    .sort_values("Modified_Unread_DTS")
)

if plot_df.empty or plot_df["RVU_actual"].isna().all():
    print("No data available for visualisation.")
else:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        plot_df["Modified_Unread_DTS"],
        plot_df["RVU_actual"],
        label="Actual RVU",
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=4,
    )
    ax.plot(
        plot_df["Modified_Unread_DTS"],
        plot_df["RVU_predicted"],
        label="Predicted RVU (3WMA)",
        color="#ff7f0e",
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=4,
    )
    ax.set_title(
        f"Hourly RVU — Actual vs 3WMA Predicted\n(prediction date: {PREDICTION_DATE_STR})",
        fontsize=13,
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total RVU")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    display(fig)
    plt.close(fig)
