"""Tools the agent can call.

Each tool = a JSON schema (what Claude sees) + a Python function (what actually runs).
The functions only READ from the AnalysisResult the pipeline already computed (plus one extra
query for cross-market CPMs). Claude never does arithmetic on raw data: it asks for numbers,
and Python computes them. Outputs use display units (%, pts, $) so the numbers the agent
quotes can be checked against them word for word.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from src.data.loader import AD_TYPES, load_market_cpm
from src.metrics.compare import weekly_changes, zscore
from src.pipeline import AnalysisResult

TYPE_ENUM = {"type": "string", "enum": AD_TYPES}


def _r(x: Any, digits: int = 2):
    """Round for JSON output; NaN -> None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), digits)


# --------------------------------------------------------------------------- tool 1
def get_network_breakdown(r: AnalysisResult, ad_type: str) -> dict:
    t = r.networks[ad_type]
    return {
        "ad_type": ad_type,
        "type_cpm_change_pct": _r(r.type_cpm_change[ad_type], 1),
        "flags": [f.reason for f in r.flags_for(ad_type)],
        "networks": [{
            "network": n,
            "share_prev_pct": _r(x["share_prev_pct"], 1), "share_this_pct": _r(x["share_this_pct"], 1),
            "share_change_pp": _r(x["share_change_pp"], 1),
            "cpm_prev_usd": _r(x["cpm_prev"]), "cpm_this_usd": _r(x["cpm_this"]),
            "cpm_change_pct": _r(x["cpm_change_pct"], 1),
            "price_effect_pts": _r(x["price_effect_pts"]), "mix_effect_pts": _r(x["mix_effect_pts"]),
        } for n, x in t.iterrows()],
        "note": "Effects are points of LTVD1 % change. Price: the network's own CPM moved. "
                "Mix: impression share moved between networks.",
    }


# --------------------------------------------------------------------------- tool 2
DAILY_METRICS = ["ltv_d1", "installs", "retention_d1", "imps_d0_per_install", "imps_d1_per_active",
                 "cpm", "network_cpm", "network_share"]


def get_daily_series(r: AnalysisResult, metric: str, ad_type: str | None = None,
                     network: str | None = None) -> dict:
    needs_type = metric in ("imps_d0_per_install", "imps_d1_per_active", "cpm", "network_cpm", "network_share")
    if needs_type and ad_type not in AD_TYPES:
        return {"error": f"metric '{metric}' needs ad_type, one of {AD_TYPES}"}
    if metric.startswith("network_") and not network:
        return {"error": f"metric '{metric}' needs a network name (see get_network_breakdown)"}

    start, end = r.prev, r.week + pd.Timedelta(days=6)
    mmp = r.data.mmp.set_index("install_date").loc[start:end]
    fb = r.data.firebase.set_index("install_date").loc[start:end]
    mon = r.data.monetization
    mon = mon[(mon["date"] >= start) & (mon["date"] <= end)]
    k = (ad_type or "").lower()
    unit, note = "", ""

    if metric == "ltv_d1":
        s, unit = mmp["rev_d1"] / mmp["installs"], "usd"
    elif metric == "installs":
        s, unit = mmp["installs"], "installs"
    elif metric == "retention_d1":
        s, unit = fb["d1_actives"] / fb["installs_fb"] * 100, "%"
        note = "By install date (the D1 return happens the next day)."
    elif metric == "imps_d0_per_install":
        s, unit = fb[f"imps_d0_{k}"] / fb["installs_fb"], "impressions"
    elif metric == "imps_d1_per_active":
        s, unit = fb[f"imps_d1_{k}"] / fb["d1_actives"], "impressions"
        note = "By install date; these impressions happen the day after the date shown."
    elif metric in ("cpm", "network_cpm", "network_share"):
        typ = mon[mon["type"] == ad_type]
        by_day = typ.groupby("date")[["impressions", "revenue"]].sum()
        if metric == "cpm":
            s, unit = by_day["revenue"] / by_day["impressions"] * 1000, "usd"
        else:
            net = typ[typ["network"] == network].groupby("date")[["impressions", "revenue"]].sum()
            net = net.reindex(by_day.index, fill_value=0)
            if metric == "network_cpm":
                s, unit = net["revenue"] / net["impressions"].where(net["impressions"] > 0) * 1000, "usd"
            else:
                s, unit = net["impressions"] / by_day["impressions"] * 100, "% of type impressions"
        note = "By calendar date (monetization table, all users)."
    else:
        return {"error": f"unknown metric '{metric}', use one of {DAILY_METRICS}"}

    digits = 4 if unit == "usd" and metric == "ltv_d1" else 2
    days = [{"date": d.date().isoformat(), "weekday": d.day_name()[:3],
             "week": "current" if d >= r.week else "previous", "value": _r(v, digits)}
            for d, v in s.items()]
    prev_vals = [x["value"] for x in days if x["week"] == "previous" and x["value"] is not None]
    this_vals = [x["value"] for x in days if x["week"] == "current" and x["value"] is not None]
    return {"metric": metric, "ad_type": ad_type, "network": network, "unit": unit, "note": note,
            "days": days,
            "previous_week_daily_mean": _r(sum(prev_vals) / len(prev_vals), digits) if prev_vals else None,
            "current_week_daily_mean": _r(sum(this_vals) / len(this_vals), digits) if this_vals else None}


# --------------------------------------------------------------------------- tool 3
TREND_METRICS = (["ltv_d1_mmp", "installs_mmp", "retention_d1"]
                 + [f"cpm_{t.lower()}" for t in AD_TYPES]
                 + [f"{t.lower()}_d0_per_install" for t in AD_TYPES]
                 + [f"{t.lower()}_d1_per_active" for t in AD_TYPES])


def get_weekly_trend(r: AnalysisResult, metric: str, weeks: int = 8) -> dict:
    if metric not in TREND_METRICS:
        return {"error": f"unknown metric '{metric}', use one of {TREND_METRICS}"}
    weeks = max(3, min(int(weeks), 14))
    kind = "pp" if metric == "retention_d1" else "pct"
    scale = 100 if metric == "retention_d1" else 1
    series = r.weekly.loc[:r.week, metric]
    changes = weekly_changes(series, kind)
    rows = [{"week": w.date().isoformat(), "value": _r(v * scale, 4 if metric.startswith("ltv") else 2),
             "wow_change": _r(changes.get(w), 1)} for w, v in series.tail(weeks).items()]
    return {"metric": metric, "unit_of_change": "pp" if kind == "pp" else "%",
            "value_unit": "%" if metric == "retention_d1" else "",
            "weeks": rows,
            "z_score_current_week": _r(zscore(series, r.week, kind, r.config["baseline_weeks"]), 1),
            "note": f"z compares the current week's change with the previous {r.config['baseline_weeks']} weekly changes."}


# --------------------------------------------------------------------------- tool 4
def compare_cpm_across_markets(r: AnalysisResult, ad_type: str) -> dict:
    df = r.market_loader(r.week, r.prev) if r.market_loader else load_market_cpm(r.source, r.week, r.prev)
    df = df[df["type"] == ad_type]
    g = df.groupby(["country", "platform", "date_week"])[["impressions", "revenue"]].sum()
    cpm = (g["revenue"] / g["impressions"] * 1000).unstack("date_week")
    if r.prev not in cpm.columns or r.week not in cpm.columns:
        return {"error": "Not enough data for both weeks."}
    change = (cpm[r.week] / cpm[r.prev] - 1) * 100
    markets = [{"market": f"{c} {p}", "cpm_change_pct": _r(v, 1),
                "is_analysed_market": (c == r.geo and p == r.platform)} for (c, p), v in change.items()]
    others = change.drop(index=(r.geo, r.platform), errors="ignore")
    return {"ad_type": ad_type, "markets": markets,
            "median_change_other_markets_pct": _r(others.median(), 1) if len(others) else None,
            "note": "If most markets moved alike, the change is likely market-wide (e.g. seasonality), "
                    "not specific to this game/market."}


# --------------------------------------------------------------------------- registry
Tool = tuple[dict, Callable[..., dict]]

TOOLS: dict[str, Tool] = {
    "get_network_breakdown": ({
        "name": "get_network_breakdown",
        "description": "Per-network impression share, CPM and their WoW changes for one ad type, with "
                       "each network's price effect and mix effect on LTVD1 (in points of LTV change), "
                       "and the flags raised for that type.",
        "input_schema": {"type": "object", "properties": {"ad_type": TYPE_ENUM}, "required": ["ad_type"]},
    }, get_network_breakdown),
    "get_daily_series": ({
        "name": "get_daily_series",
        "description": "Daily values over the previous and current week (14 days). Use it to tell a step "
                       "change on a specific day (release, SDK or network issue) from a gradual drift.",
        "input_schema": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": DAILY_METRICS},
            "ad_type": {**TYPE_ENUM, "description": "Required for impression, CPM and network metrics."},
            "network": {"type": "string", "description": "Required for network_cpm and network_share."},
        }, "required": ["metric"]},
    }, get_daily_series),
    "get_weekly_trend": ({
        "name": "get_weekly_trend",
        "description": "Weekly values and week-over-week changes for one metric, plus the z-score of the "
                       "current week. Use it to see whether a change is new, part of a trend, or noise.",
        "input_schema": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": TREND_METRICS},
            "weeks": {"type": "integer", "minimum": 3, "maximum": 14, "description": "Default 8."},
        }, "required": ["metric"]},
    }, get_weekly_trend),
    "compare_cpm_across_markets": ({
        "name": "compare_cpm_across_markets",
        "description": "WoW CPM change for one ad type in every geo/platform. Use it to check whether a CPM "
                       "move is market-wide (seasonality, demand) or specific to the analysed market.",
        "input_schema": {"type": "object", "properties": {"ad_type": TYPE_ENUM}, "required": ["ad_type"]},
    }, compare_cpm_across_markets),
}


def execute(r: AnalysisResult, name: str, inputs: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {list(TOOLS)} and submit_findings."}
    try:
        return TOOLS[name][1](r, **inputs)
    except Exception as e:  # a tool error goes back to the agent, it doesn't crash the run
        return {"error": f"{type(e).__name__}: {e}"}


def submit_findings_schema(driver_names: list[str]) -> dict:
    """The final 'answer' tool. Driver names are an enum built from this week's drivers table,
    so the agent can only attribute the change to drivers that actually exist."""
    return {
        "name": "submit_findings",
        "description": "Submit the final analysis. Call this exactly once, when the investigation is done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "One sentence: what happened to LTVD1 and the main reason."},
                "summary": {"type": "string", "description": "2-5 sentences for a growth/monetization lead."},
                "drivers": {
                    "type": "array", "maxItems": 5,
                    "items": {"type": "object", "properties": {
                        "driver": {"type": "string", "enum": driver_names},
                        "network": {"type": "string", "description": "Optional: the network responsible."},
                        "explanation": {"type": "string", "description": "What the evidence shows and the likely cause."},
                        "verdict": {"type": "string", "enum": ["specific_issue", "market_wide", "normal_variation"],
                                    "description": "specific_issue: a real change in this game/market that someone should "
                                                   "act on or investigate. market_wide: real, but shared with other markets "
                                                   "(seasonality, demand). normal_variation: within normal weekly noise."},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    }, "required": ["driver", "explanation", "verdict", "confidence"]},
                },
                "next_steps": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                "caveats": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
            },
            "required": ["headline", "summary", "drivers", "next_steps"],
        },
    }
