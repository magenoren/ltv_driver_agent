"""Integration tests: generate the synthetic world, run the metrics, and check that the
planted scenarios (synthetic.yaml) show up where they should - and nowhere else.

The last test compares BigQuery (SQL shaping) with the CSV path (pandas shaping). It only runs
when you ask for it, because it needs BigQuery access:
    $env:RUN_BQ_TESTS = "1"; pytest -v      (PowerShell)
"""
import os

import pandas as pd
import pytest

from src.data import quality
from src.data.loader import from_raw, load
from src.data.synthetic import generate
from src.metrics.compare import change, source_validation, zscore
from src.metrics.monetization import network_breakdown
from src.metrics.weekly import build_weekly
from src.settings import load_yaml

WEEK = pd.Timestamp("2026-11-01")
PREV = WEEK - pd.Timedelta(days=7)
START = WEEK - pd.Timedelta(weeks=13)


@pytest.fixture(scope="module")
def raw():
    return generate(load_yaml("synthetic.yaml"))


def run(raw, geo, platform):
    data = from_raw(raw, geo, platform, START, WEEK)
    return data, build_weekly(data)


def wow(weekly, column, kind="pct"):
    return change(weekly.at[WEEK, column], weekly.at[PREV, column], kind)


def test_data_quality_passes(raw):
    data, _ = run(raw, "US", "ios")
    assert quality.check(data, WEEK, PREV) == []


def test_quality_catches_missing_day(raw):
    data, _ = run(raw, "US", "ios")
    data.mmp = data.mmp[data.mmp["install_date"] != pd.Timestamp("2026-11-03")]
    assert any("MMP" in issue and "6/7" in issue for issue in quality.check(data, WEEK, PREV))


def test_us_ios_interstitial_d0_drop(raw):
    _, weekly = run(raw, "US", "ios")
    assert wow(weekly, "interstitial_d0_per_install") < -12          # planted: -18%
    assert zscore(weekly["interstitial_d0_per_install"], WEEK, "pct", 12) < -3


def test_gb_android_network_effects(raw):
    data, _ = run(raw, "GB", "android")
    inter = network_breakdown(data.monetization, "Interstitial", WEEK, PREV)
    assert inter.at["Mintegral", "cpm_change_pct"] > 35             # planted: +45%
    assert abs(inter.at["AppLovin", "cpm_change_pct"]) < 10          # others: seasonality + noise only
    rewarded = network_breakdown(data.monetization, "Rewarded", WEEK, PREV)
    assert rewarded.at["AdMob", "share_this_pct"] == 0               # planted: AdMob stopped


def test_us_android_retention_drop(raw):
    _, weekly = run(raw, "US", "android")
    assert wow(weekly, "retention_d1", "pp") < -1.5                  # planted: -13% from Tuesday


@pytest.mark.parametrize("geo,platform", [("GB", "ios"), ("DE", "ios"), ("DE", "android"), ("BR", "ios")])
def test_controls_have_no_extreme_ltv_move(raw, geo, platform):
    _, weekly = run(raw, geo, platform)
    assert abs(zscore(weekly["ltv_d1_mmp"], WEEK, "pct", 12)) < 3


@pytest.mark.parametrize("geo,platform", [("US", "ios"), ("US", "android"), ("GB", "android"),
                                          ("GB", "ios"), ("DE", "ios"), ("BR", "android")])
def test_sources_agree(raw, geo, platform):
    _, weekly = run(raw, geo, platform)
    assert not source_validation(weekly, WEEK, threshold_pp=2)["flagged"]


@pytest.mark.skipif(os.getenv("RUN_BQ_TESTS") != "1", reason="set RUN_BQ_TESTS=1 to run against BigQuery")
def test_bigquery_sql_matches_pandas():
    bq = load("bq", "US", "ios", START, WEEK)
    csv = load("csv", "US", "ios", START, WEEK)
    for name in ("mmp", "firebase", "monetization"):
        a = getattr(bq, name).reset_index(drop=True)
        b = getattr(csv, name).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b[a.columns], check_dtype=False, check_exact=False, rtol=1e-6)
