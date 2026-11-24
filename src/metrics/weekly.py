"""Weekly metrics: one row per week, one column per metric.

Rule used throughout: sum the numerators and denominators over the week first, then divide.
Averaging seven daily ratios would give a Tuesday with few installs the same weight as a
busy Saturday - a classic source of small but confusing discrepancies.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import AD_TYPES, SourceData
from src.metrics.monetization import cpm_by_type


def cohort_metrics(mmp: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    m = mmp.groupby("date_week")[["installs", "rev_d0", "rev_d1"]].sum()
    f = fb.groupby("date_week").sum(numeric_only=True)

    out = pd.DataFrame(index=m.index.union(f.index))
    out["installs_mmp"] = m["installs"]
    out["installs_fb"] = f["installs_fb"]
    # LTV (MMP): revenue per install. rev_d1 is already cumulative D0+D1.
    out["ltv_d0_mmp"] = m["rev_d0"] / m["installs"]
    out["ltv_d1_mmp"] = m["rev_d1"] / m["installs"]
    # LTV (Firebase): ad revenue events, divided by D0 active users as the install count
    out["ltv_d0_fb"] = f["ad_rev_d0"] / f["installs_fb"]
    out["ltv_d1_fb"] = (f["ad_rev_d0"] + f["ad_rev_d1"]) / f["installs_fb"]
    out["retention_d1"] = f["d1_actives"] / f["installs_fb"]

    for t in AD_TYPES:
        k = t.lower()
        d0, d1 = f[f"imps_d0_{k}"], f[f"imps_d1_{k}"]
        out[f"{k}_d01_per_install"] = (d0 + d1) / f["installs_fb"]  # your summary definition
        out[f"{k}_d0_per_install"] = d0 / f["installs_fb"]            # used by the decomposition
        out[f"{k}_d1_per_active"] = d1 / f["d1_actives"]              # D1 engagement, net of retention
    return out


def build_weekly(data: SourceData) -> pd.DataFrame:
    """All weekly metrics in one table, indexed by date_week."""
    weekly = cohort_metrics(data.mmp, data.firebase)
    cpm = cpm_by_type(data.monetization)
    cpm.columns = [f"cpm_{t.lower()}" for t in cpm.columns]
    return weekly.join(cpm, how="left").sort_index()
