"""Load the three sources for one geo/platform, from BigQuery or from the local CSVs.

Both paths return the same three "daily" DataFrames, so everything downstream doesn't care
where the data came from:
  mmp           install_date, date_week, installs, rev_d0, rev_d1
  firebase      install_date, date_week, installs_fb, d1_actives, ad_rev_d0, ad_rev_d1,
                imps_d0_<type>, imps_d1_<type>
  monetization  date, date_week, network, type, impressions, revenue

In BigQuery the shaping is done in SQL (src/data/queries/*.sql). For CSVs, the shape_* functions
below do the same in pandas. A test checks that both give identical results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.settings import PROJECT_ROOT

QUERY_DIR = Path(__file__).parent / "queries"
CSV_DIR = PROJECT_ROOT / "data" / "synthetic"
AD_TYPES = ["Banner", "Interstitial", "Rewarded"]


@dataclass
class SourceData:
    mmp: pd.DataFrame
    firebase: pd.DataFrame
    monetization: pd.DataFrame


# --------------------------------------------------------------------------- shaping (pandas)
def shape_mmp(raw: pd.DataFrame) -> pd.DataFrame:
    out = (raw.groupby(["install_date", "date_week"], as_index=False)
              [["installs", "cohort_rev_d0", "cohort_rev_d1"]].sum()
              .rename(columns={"cohort_rev_d0": "rev_d0", "cohort_rev_d1": "rev_d1"}))
    return out.sort_values("install_date").reset_index(drop=True)


def shape_firebase(raw: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of firebase_daily.sql: long (age x type) rows -> one wide row per install date."""
    raw = raw[raw["age"] <= 1]
    keys = ["install_date", "date_week"]

    def pick(type_: str, age: int, column: str, name: str) -> pd.Series:
        rows = raw[(raw["type"] == type_) & (raw["age"] == age)]
        return rows.groupby(keys)[column].sum().rename(name)

    parts = [pick("other", 0, "users", "installs_fb"),
             pick("other", 1, "users", "d1_actives"),
             pick("ad_revenue", 0, "value", "ad_rev_d0"),
             pick("ad_revenue", 1, "value", "ad_rev_d1")]
    for age in (0, 1):
        for t in AD_TYPES:
            parts.append(pick(t, age, "events", f"imps_d{age}_{t.lower()}"))
    out = pd.concat(parts, axis=1).fillna(0).reset_index()
    return out.sort_values("install_date").reset_index(drop=True)


def shape_monetization(raw: pd.DataFrame) -> pd.DataFrame:
    out = (raw.groupby(["date", "date_week", "network", "type"], as_index=False)
              [["impressions", "revenue"]].sum())
    return out.sort_values(["date", "type", "network"]).reset_index(drop=True)


def _to_timestamps(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Normalise types where data enters the project, so both sources look identical downstream.

    - Dates -> pandas Timestamps (BigQuery returns DATE as its own 'dbdate' type).
    - Numbers -> plain float64. BigQuery returns *nullable* types (Int64, Float64), where a
      missing value is pd.NA instead of NaN. pd.NA can't be used in an `if` ("boolean value of
      NA is ambiguous") and spreads through arithmetic differently, so we convert once, here.
    - Text -> plain str.
    """
    df = df.copy()
    for c in df.columns:
        if c in cols:
            df[c] = pd.to_datetime(df[c].astype(str))
        elif pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64")
        else:
            df[c] = df[c].astype(str)
    return df


# --------------------------------------------------------------------------- CSV source
def _read_csv(name: str, geo: str, platform: str) -> pd.DataFrame:
    path = CSV_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python generate_data.py --no-upload")
    df = pd.read_csv(path)
    return df[(df["country"] == geo) & (df["platform"] == platform)].copy()


def from_raw(raw: dict[str, pd.DataFrame], geo: str, platform: str,
             start_week: pd.Timestamp, end_week: pd.Timestamp) -> SourceData:
    """Raw source tables (as generated / as in the CSVs) -> SourceData for one geo/platform."""
    def select(df):
        df = df[(df["country"] == geo) & (df["platform"] == platform)]
        week = pd.to_datetime(df["date_week"].astype(str))
        return df[(week >= start_week) & (week <= end_week)]

    mmp = shape_mmp(select(raw["mmp"]))
    fb = shape_firebase(select(raw["firebase"]))
    mon = shape_monetization(select(raw["monetization"]))
    return SourceData(_to_timestamps(mmp, ["install_date", "date_week"]),
                      _to_timestamps(fb, ["install_date", "date_week"]),
                      _to_timestamps(mon, ["date", "date_week"]))


def _load_csv(geo, platform, start_week, end_week) -> SourceData:
    raw = {name: _read_csv(name, geo, platform) for name in ("mmp", "firebase", "monetization")}
    return from_raw(raw, geo, platform, start_week, end_week)


# --------------------------------------------------------------------------- BigQuery source
def _run_query(client, sql_file: str, table_id: str, params: dict) -> pd.DataFrame:
    from google.cloud import bigquery

    sql = (QUERY_DIR / sql_file).read_text(encoding="utf-8").format(table=table_id)
    types = {str: "STRING", pd.Timestamp: "DATE"}
    query_params = [
        bigquery.ScalarQueryParameter(name, types[type(value)],
                                      value.date() if isinstance(value, pd.Timestamp) else value)
        for name, value in params.items()
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    # REST download: right for small results, and avoids the optional BigQuery Storage package
    return client.query(sql, job_config=job_config).to_dataframe(create_bqstorage_client=False)


def _load_bq(geo, platform, start_week, end_week) -> SourceData:
    from src.data.bigquery import get_client
    from src.settings import get_bigquery_settings

    s = get_bigquery_settings()
    client = get_client(s)
    p = {"country": geo, "platform": platform, "start_week": start_week, "end_week": end_week}
    mmp = _run_query(client, "mmp_daily.sql", s.mmp_id, p)
    fb = _run_query(client, "firebase_daily.sql", s.firebase_id, p)
    mon = _run_query(client, "monetization_daily.sql", s.monetization_id, p)
    return SourceData(_to_timestamps(mmp, ["install_date", "date_week"]),
                      _to_timestamps(fb, ["install_date", "date_week"]),
                      _to_timestamps(mon, ["date", "date_week"]))


# --------------------------------------------------------------------------- public API
def latest_complete_week(source: str, geo: str, platform: str) -> pd.Timestamp:
    """Most recent week with all 7 install dates in the MMP table."""
    if source == "csv":
        mmp = _read_csv("mmp", geo, platform)
        days = mmp.groupby("date_week")["install_date"].nunique()
        complete = days[days == 7]
        if complete.empty:
            raise ValueError(f"No complete week for {geo} {platform}")
        return pd.Timestamp(complete.index.max())

    from src.data.bigquery import get_client
    from src.settings import get_bigquery_settings

    s = get_bigquery_settings()
    df = _run_query(get_client(s), "latest_complete_week.sql", s.mmp_id,
                    {"country": geo, "platform": platform})
    if df.empty or pd.isna(df.loc[0, "week"]):
        raise ValueError(f"No complete week for {geo} {platform}")
    return pd.Timestamp(str(df.loc[0, "week"]))


def load_market_cpm(source: str, week: pd.Timestamp, prev: pd.Timestamp) -> pd.DataFrame:
    """Weekly impressions/revenue per geo x platform x type, all markets, two weeks."""
    if source == "csv":
        path = CSV_DIR / "monetization.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run: python generate_data.py --no-upload")
        raw = pd.read_csv(path)
        raw = raw[pd.to_datetime(raw["date_week"]).isin([prev, week])]
        df = raw.groupby(["date_week", "country", "platform", "type"], as_index=False)[
            ["impressions", "revenue"]].sum()
    else:
        from src.data.bigquery import get_client
        from src.settings import get_bigquery_settings

        s = get_bigquery_settings()
        df = _run_query(get_client(s), "market_cpm.sql", s.monetization_id, {"week": week, "prev_week": prev})
    return _to_timestamps(df, ["date_week"])


def load(source: str, geo: str, platform: str,
         start_week: pd.Timestamp, end_week: pd.Timestamp) -> SourceData:
    if source == "csv":
        return _load_csv(geo, platform, start_week, end_week)
    if source == "bq":
        return _load_bq(geo, platform, start_week, end_week)
    raise ValueError(f"Unknown source '{source}' (use 'bq' or 'csv')")
