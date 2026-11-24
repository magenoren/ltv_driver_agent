"""The deterministic core: every number in the report (and later every number the agent sees)
comes from run(). One function, one result object - so the console, the HTML report and the
agent can't drift apart."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.data import quality
from src.data.loader import AD_TYPES, SourceData, latest_complete_week, load
from src.metrics.compare import (ENGAGEMENT_METRICS, SUMMARY_METRICS, source_validation,
                                 summary_table, zscore)
from src.metrics.decomposition import Decomposition, decompose, driver_networks, driver_table
from src.metrics.flags import Flag, network_flags
from src.metrics.monetization import network_breakdown
from src.metrics.weekly import build_weekly
from src.settings import load_yaml


@dataclass
class AnalysisResult:
    geo: str
    platform: str
    week: pd.Timestamp
    prev: pd.Timestamp
    source: str
    config: dict
    data: SourceData
    weekly: pd.DataFrame
    quality_issues: list[str]
    summary: pd.DataFrame
    engagement: pd.DataFrame
    validation: dict
    decomposition: Decomposition
    drivers: pd.DataFrame
    ltv_zscore: float
    networks: dict[str, pd.DataFrame] = field(default_factory=dict)   # per ad type
    type_cpm_change: dict[str, float] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    # Optional: where compare_cpm_across_markets gets all-market data (the evaluation injects
    # in-memory data here; normally it queries the same source as the rest of the run)
    market_loader: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None

    def flags_for(self, ad_type: str) -> list[Flag]:
        return [f for f in self.flags if f.ad_type == ad_type]


def resolve_week(week: str | pd.Timestamp | None, source: str, geo: str, platform: str) -> pd.Timestamp:
    w = pd.Timestamp(week) if week is not None else latest_complete_week(source, geo, platform)
    if w.dayofweek != 6:  # pandas: Monday=0 ... Sunday=6
        raise ValueError(f"Week must start on a Sunday; {w.date()} is a {w.day_name()}")
    return w


def analyze(data: SourceData, geo: str, platform: str, week: pd.Timestamp, source: str,
            config: dict) -> AnalysisResult:
    th, n = config["thresholds"], config["baseline_weeks"]
    prev = week - pd.Timedelta(days=7)

    weekly = build_weekly(data)
    dec = decompose(weekly, data.monetization, week)
    result = AnalysisResult(
        geo=geo, platform=platform, week=week, prev=prev, source=source, config=config,
        data=data, weekly=weekly,
        quality_issues=quality.check(data, week, prev),
        summary=summary_table(weekly, week, SUMMARY_METRICS, n),
        engagement=summary_table(weekly, week, ENGAGEMENT_METRICS, n),
        validation=source_validation(weekly, week, th["source_gap_pp"]),
        decomposition=dec,
        drivers=driver_table(dec, weekly, th["share_of_change_min_pct"],
                             th.get("share_of_change_min_concentration", 0.5)),
        ltv_zscore=zscore(weekly["ltv_d1_mmp"], week, "pct", n),
    )
    for t in AD_TYPES:
        col = f"cpm_{t.lower()}"
        result.type_cpm_change[t] = (weekly.at[week, col] / weekly.at[prev, col] - 1) * 100
        nb = network_breakdown(data.monetization, t, week, prev, th["network_min_share_pct"])
        result.networks[t] = driver_networks(dec, t, th["network_min_share_pct"], nb)
        result.flags += network_flags(t, result.type_cpm_change[t], result.networks[t], th)
    return result


def run(geo: str, platform: str, week: str | None = None, source: str = "bq") -> AnalysisResult:
    geo, platform = geo.upper(), platform.lower()
    config = load_yaml("config.yaml")
    w = resolve_week(week, source, geo, platform)
    start = w - pd.Timedelta(weeks=config["baseline_weeks"] + 1)
    data = load(source, geo, platform, start, w)
    return analyze(data, geo, platform, w, source, config)
