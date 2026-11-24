"""BigQuery helpers: client, explicit table schemas, upload and a sanity-check query.

Authentication uses Application Default Credentials (ADC): run once
    gcloud auth application-default login
and the client library finds your credentials automatically. No key files in the code.
"""
from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from src.settings import BigQuerySettings

F = bigquery.SchemaField

# Explicit schemas: we decide the column types instead of letting BigQuery guess.
# This prevents surprises like dates stored as strings or counts stored as floats.
SCHEMAS: dict[str, list[bigquery.SchemaField]] = {
    "mmp": [
        F("install_date", "DATE", mode="REQUIRED"),
        F("date_week", "DATE", mode="REQUIRED", description="Sunday-based week of install_date"),
        F("platform", "STRING", mode="REQUIRED"),
        F("country", "STRING", mode="REQUIRED"),
        F("installs", "INT64"),
        F("cohort_rev_d0", "FLOAT64", description="Cohort revenue on day 0"),
        F("cohort_rev_d1", "FLOAT64", description="Cumulative cohort revenue, days 0-1"),
    ],
    "firebase": [
        F("install_date", "DATE", mode="REQUIRED"),
        F("date_week", "DATE", mode="REQUIRED"),
        F("platform", "STRING", mode="REQUIRED"),
        F("country", "STRING", mode="REQUIRED"),
        F("age", "INT64", mode="REQUIRED", description="Days since install"),
        F("type", "STRING", mode="REQUIRED",
          description="Banner | Interstitial | Rewarded | other | ad_revenue"),
        F("users", "INT64", description="Users with at least one event of this type"),
        F("events", "INT64"),
        F("value", "FLOAT64", description="Revenue for ad_revenue rows; equals events otherwise"),
    ],
    "monetization": [
        F("date", "DATE", mode="REQUIRED"),
        F("date_week", "DATE", mode="REQUIRED"),
        F("platform", "STRING", mode="REQUIRED"),
        F("country", "STRING", mode="REQUIRED"),
        F("network", "STRING", mode="REQUIRED"),
        F("type", "STRING", mode="REQUIRED"),
        F("impressions", "INT64"),
        F("revenue", "FLOAT64"),
    ],
}

# Clustering sorts the stored data by these columns, so queries filtered on one geo/platform
# scan less data. (We avoid date partitioning: in the free BigQuery sandbox, partitions older
# than 60 days expire automatically, which would delete most of our history.)
CLUSTERING = ["country", "platform"]


def get_client(settings: BigQuerySettings) -> bigquery.Client:
    return bigquery.Client(project=settings.project, location=settings.location)


def ensure_dataset(client: bigquery.Client, settings: BigQuerySettings) -> None:
    dataset = bigquery.Dataset(f"{settings.project}.{settings.dataset}")
    dataset.location = settings.location
    client.create_dataset(dataset, exists_ok=True)  # no-op if it already exists


def upload_dataframe(client: bigquery.Client, df: pd.DataFrame, table_id: str,
                     schema: list[bigquery.SchemaField]) -> int:
    """Load a DataFrame into a table, replacing its contents. Returns the row count."""
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # rerunning = same result (idempotent)
        clustering_fields=CLUSTERING,
    )
    # A load job (not streaming inserts): free, and allowed in the sandbox.
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # wait for the job to finish; raises if it failed
    return client.get_table(table_id).num_rows


def sanity_check(client: bigquery.Client, settings: BigQuerySettings) -> pd.DataFrame:
    """Weekly LTVD1 from both cohort tables for the last 3 weeks, US iOS: proves the upload works
    and gives you a first BigQuery SQL query to read."""
    sql = f"""
    WITH mmp AS (
      SELECT date_week,
             SUM(installs)                          AS installs_mmp,
             SAFE_DIVIDE(SUM(cohort_rev_d1), SUM(installs)) AS ltv_d1_mmp
      FROM `{settings.mmp_id}`
      WHERE country = @country AND platform = @platform
      GROUP BY date_week
    ),
    fb AS (
      SELECT date_week,
             SUM(IF(type = 'other' AND age = 0, users, 0))       AS installs_fb,
             SAFE_DIVIDE(SUM(IF(type = 'ad_revenue', value, 0)),
                         SUM(IF(type = 'other' AND age = 0, users, 0))) AS ltv_d1_fb
      FROM `{settings.firebase_id}`
      WHERE country = @country AND platform = @platform AND age <= 1
      GROUP BY date_week
    )
    SELECT * FROM mmp JOIN fb USING (date_week)
    ORDER BY date_week DESC
    LIMIT 3
    """
    # Query parameters (@country) instead of pasting values into the string: safer and cacheable.
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("country", "STRING", "US"),
        bigquery.ScalarQueryParameter("platform", "STRING", "ios"),
    ])
    # REST download: right for small results, and avoids the optional BigQuery Storage package
    return client.query(sql, job_config=job_config).to_dataframe(create_bqstorage_client=False)
