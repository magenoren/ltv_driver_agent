"""Unit tests on tiny hand-made data, where the right answer can be worked out by hand.
Run from the project root:  pytest -v
"""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import shape_firebase
from src.metrics.compare import change, source_validation, zscore
from src.metrics.monetization import network_breakdown
from src.metrics.weekly import cohort_metrics

W1, W2 = pd.Timestamp("2026-10-25"), pd.Timestamp("2026-11-01")


def fb_day(date, week, installs, d1, rev0, rev1, inter0=0, inter1=0):
    row = {"install_date": pd.Timestamp(date), "date_week": week, "installs_fb": installs,
           "d1_actives": d1, "ad_rev_d0": rev0, "ad_rev_d1": rev1}
    for t in ("banner", "interstitial", "rewarded"):
        row[f"imps_d0_{t}"] = inter0 if t == "interstitial" else 0
        row[f"imps_d1_{t}"] = inter1 if t == "interstitial" else 0
    return row


def mmp_day(date, week, installs, rev0, rev1):
    return {"install_date": pd.Timestamp(date), "date_week": week,
            "installs": installs, "rev_d0": rev0, "rev_d1": rev1}


def test_cohort_metric_formulas():
    fb = pd.DataFrame([fb_day("2026-11-01", W2, 1000, 300, 100.0, 50.0, inter0=5000, inter1=2100)])
    mmp = pd.DataFrame([mmp_day("2026-11-01", W2, 1040, 104.0, 156.0)])
    m = cohort_metrics(mmp, fb).loc[W2]
    assert m["ltv_d1_fb"] == pytest.approx(0.15)            # (100 + 50) / 1000
    assert m["ltv_d0_fb"] == pytest.approx(0.10)
    assert m["ltv_d1_mmp"] == pytest.approx(0.15)           # 156 / 1040
    assert m["retention_d1"] == pytest.approx(0.30)
    assert m["interstitial_d01_per_install"] == pytest.approx(7.1)   # (5000 + 2100) / 1000
    assert m["interstitial_d0_per_install"] == pytest.approx(5.0)
    assert m["interstitial_d1_per_active"] == pytest.approx(7.0)     # 2100 / 300


def test_weekly_ratio_is_sum_over_sum_not_average_of_days():
    mmp = pd.DataFrame([mmp_day("2026-11-01", W2, 100, 5.0, 10.0),     # LTV 0.10
                        mmp_day("2026-11-02", W2, 900, 90.0, 180.0)])   # LTV 0.20
    fb = pd.DataFrame([fb_day("2026-11-01", W2, 100, 30, 5, 5), fb_day("2026-11-02", W2, 900, 270, 90, 90)])
    # Install-weighted: 190 / 1000 = 0.19. Averaging the days would wrongly give 0.15.
    assert cohort_metrics(mmp, fb).at[W2, "ltv_d1_mmp"] == pytest.approx(0.19)


def test_shape_firebase_pivots_long_rows():
    rows = [
        ("other", 0, 1000, 30000, 30000.0), ("other", 1, 350, 7700, 7700.0),
        ("Interstitial", 0, 750, 6000, 6000.0), ("Interstitial", 1, 260, 2450, 2450.0),
        ("Banner", 0, 920, 60000, 60000.0), ("Rewarded", 0, 400, 2500, 2500.0),
        ("ad_revenue", 0, 920, 68500, 210.5), ("ad_revenue", 1, 330, 2450, 45.25),
        ("other", 2, 200, 4000, 4000.0),  # age 2 must be ignored
    ]
    raw = pd.DataFrame([{"install_date": "2026-11-01", "date_week": "2026-11-01", "age": age,
                         "type": t, "users": u, "events": e, "value": v} for t, age, u, e, v in rows])
    out = shape_firebase(raw).iloc[0]
    assert out["installs_fb"] == 1000 and out["d1_actives"] == 350
    assert out["ad_rev_d0"] == pytest.approx(210.5) and out["ad_rev_d1"] == pytest.approx(45.25)
    assert out["imps_d0_interstitial"] == 6000 and out["imps_d1_interstitial"] == 2450
    assert out["imps_d1_banner"] == 0  # missing combination -> 0, not NaN


def test_change_pct_vs_pp():
    assert change(0.33, 0.30, "pct") == pytest.approx(10.0)   # relative
    assert change(0.33, 0.30, "pp") == pytest.approx(3.0)     # percentage points


def test_zscore():
    # Build a series whose weekly changes are +1%, -1%, ... then a +5% week
    changes = [1, -1, 1, -1, 1, -1, 5]
    values = np.cumprod([100.0] + [1 + c / 100 for c in changes])
    weeks = pd.date_range("2026-09-06", periods=len(values), freq="7D")
    series = pd.Series(values, index=weeks)
    expected = (5 - 0) / np.std([1, -1, 1, -1, 1, -1], ddof=1)
    assert zscore(series, weeks[-1], "pct", n_weeks=12) == pytest.approx(expected)
    assert np.isnan(zscore(series, weeks[2], "pct", n_weeks=12))  # too little history


def test_network_breakdown_groups_small_and_keeps_stopped_networks():
    rows = [  # (week, network, impressions, revenue)
        (W1, "A", 600, 6.00), (W1, "B", 370, 5.55), (W1, "C", 30, 0.15),
        (W2, "A", 980, 11.76), (W2, "C", 20, 0.10),          # B stopped serving
    ]
    mon = pd.DataFrame([{"date": w, "date_week": w, "network": n, "type": "Rewarded",
                         "impressions": i, "revenue": r} for w, n, i, r in rows])
    nb = network_breakdown(mon, "Rewarded", W2, W1, min_share_pct=5)
    assert list(nb.index) == ["A", "B", "Other"]             # C (<5% both weeks) -> Other
    assert nb.at["A", "cpm_change_pct"] == pytest.approx(20.0)  # $10 -> $12
    assert nb.at["B", "share_this_pct"] == 0
    assert np.isnan(nb.at["B", "cpm_this"])                  # no impressions -> no CPM
    assert nb["share_this_pct"].sum() == pytest.approx(100)


def test_source_validation_flag():
    weekly = pd.DataFrame({"ltv_d1_mmp": [0.30, 0.27], "ltv_d1_fb": [0.29, 0.29]}, index=[W1, W2])
    v = source_validation(weekly, W2, threshold_pp=2)
    assert v["mmp_wow_pct"] == pytest.approx(-10.0)
    assert v["fb_wow_pct"] == pytest.approx(0.0)
    assert v["flagged"]


def test_loader_normalises_bigquery_nullable_types():
    """BigQuery returns Int64/Float64 (nullable) and dbdate columns; downstream code expects
    plain float64 and Timestamps. This mimics a BigQuery result and checks the conversion."""
    from src.data.loader import _to_timestamps
    bq_like = pd.DataFrame({
        "date": pd.array(["2026-11-01", "2026-11-02"], dtype="string"),
        "date_week": pd.array(["2026-11-01", "2026-11-01"], dtype="string"),
        "network": pd.array(["AdMob", "Unity"], dtype="string"),
        "impressions": pd.array([100, None], dtype="Int64"),
        "revenue": pd.array([1.5, None], dtype="Float64"),
    })
    out = _to_timestamps(bq_like, ["date", "date_week"])
    assert out["impressions"].dtype == "float64" and out["revenue"].dtype == "float64"
    assert np.isnan(out.loc[1, "impressions"])          # pd.NA became NaN
    assert str(out["date"].dtype).startswith("datetime64")
    assert pd.api.types.is_string_dtype(out["network"])   # "object" in pandas 2, "str" in pandas 3
