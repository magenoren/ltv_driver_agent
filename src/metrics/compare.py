"""Week-over-week comparisons.

Two ideas here:
1. % vs pp: ratios like LTV or CPM change in percent; rates like retention change in
   percentage points (35.0% -> 33.6% is -1.4pp, which is a -4% relative change).
2. "Is this change unusual?": a -4% move means little if this metric routinely moves +-5% a week.
   We compare this week's change to the distribution of the previous N weekly changes (z-score).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    fmt: str          # how to print the level: "usd", "usd4", "pct", "num", "int"
    change: str       # "pct" (relative %) or "pp" (percentage points)


SUMMARY_METRICS = [
    MetricSpec("ltv_d0_mmp", "LTV D0 (MMP)", "usd4", "pct"),
    MetricSpec("ltv_d0_fb", "LTV D0 (Firebase)", "usd4", "pct"),
    MetricSpec("ltv_d1_mmp", "LTV D1 (MMP)", "usd4", "pct"),
    MetricSpec("ltv_d1_fb", "LTV D1 (Firebase)", "usd4", "pct"),
    MetricSpec("installs_mmp", "Installs (MMP)", "int", "pct"),
    MetricSpec("retention_d1", "Retention D1", "pct", "pp"),
    MetricSpec("interstitial_d01_per_install", "Interstitials per install, D0-1", "num", "pct"),
    MetricSpec("rewarded_d01_per_install", "Rewardeds per install, D0-1", "num", "pct"),
    MetricSpec("banner_d01_per_install", "Banners per install, D0-1", "num", "pct"),
    MetricSpec("cpm_interstitial", "Interstitial CPM", "usd", "pct"),
    MetricSpec("cpm_rewarded", "Rewarded CPM", "usd", "pct"),
    MetricSpec("cpm_banner", "Banner CPM", "usd", "pct"),
]

ENGAGEMENT_METRICS = [
    MetricSpec(f"{t}_{kind}", f"{t.capitalize()} {label}", "num", "pct")
    for t in ("interstitial", "rewarded", "banner")
    for kind, label in (("d0_per_install", "D0 per install"), ("d1_per_active", "D1 per D1 active"))
]


def change(this: float | pd.Series, prev: float | pd.Series, kind: str):
    if kind == "pct":
        return (this / prev - 1) * 100
    if kind == "pp":
        return (this - prev) * 100
    raise ValueError(kind)


def weekly_changes(series: pd.Series, kind: str) -> pd.Series:
    """Change vs the previous week, for every week in the series."""
    return change(series, series.shift(1), kind)


def zscore(series: pd.Series, week: pd.Timestamp, kind: str, n_weeks: int) -> float:
    """How unusual is this week's change, compared with the previous n_weeks changes?
    Returns NaN when there is too little history to judge."""
    changes = weekly_changes(series, kind).dropna()
    if week not in changes.index:
        return np.nan
    history = changes[changes.index < week].tail(n_weeks)
    if len(history) < 4 or history.std(ddof=1) == 0:
        return np.nan
    return float((changes[week] - history.mean()) / history.std(ddof=1))


def summary_table(weekly: pd.DataFrame, week: pd.Timestamp, specs: list[MetricSpec],
                  baseline_weeks: int) -> pd.DataFrame:
    prev = week - pd.Timedelta(days=7)
    rows = []
    for s in specs:
        this_v, prev_v = weekly.at[week, s.column], weekly.at[prev, s.column]
        rows.append({
            "metric": s.label,
            "column": s.column,
            "fmt": s.fmt,
            "prev_week": prev_v,
            "this_week": this_v,
            "change": change(this_v, prev_v, s.change),
            "change_unit": "%" if s.change == "pct" else "pp",
            "z_score": zscore(weekly[s.column], week, s.change, baseline_weeks),
        })
    return pd.DataFrame(rows)


def source_validation(weekly: pd.DataFrame, week: pd.Timestamp, threshold_pp: float) -> dict:
    """Do MMP and Firebase tell the same story? Levels differ structurally (install counting,
    timezones), so the flag is about the WoW *changes* disagreeing, with the level gap reported."""
    prev = week - pd.Timedelta(days=7)
    mmp_wow = change(weekly.at[week, "ltv_d1_mmp"], weekly.at[prev, "ltv_d1_mmp"], "pct")
    fb_wow = change(weekly.at[week, "ltv_d1_fb"], weekly.at[prev, "ltv_d1_fb"], "pct")
    gap = (weekly["ltv_d1_mmp"] / weekly["ltv_d1_fb"] - 1) * 100
    return {
        "level_gap_this_pct": float(gap[week]),
        "level_gap_prev_pct": float(gap[prev]),
        "mmp_wow_pct": float(mmp_wow),
        "fb_wow_pct": float(fb_wow),
        "wow_diff_pp": float(mmp_wow - fb_wow),
        "flagged": bool(abs(mmp_wow - fb_wow) > threshold_pp),
        "threshold_pp": threshold_pp,
    }
