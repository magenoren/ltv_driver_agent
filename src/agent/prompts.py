"""What the agent is told.

SYSTEM_PROMPT: the role, the method and the rules - the same for every run.
build_briefing(): this week's core results, computed by the pipeline. The agent starts from
these instead of from raw data, so its first tool call is already a targeted question.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from src.pipeline import AnalysisResult

SYSTEM_PROMPT = """You are a senior monetization analyst for a free-to-play mobile game that earns \
almost all of its revenue from ads (banner, interstitial, rewarded). Every week you explain what \
changed LTVD1 (revenue per install over the install day and the next day) for one market (geo + \
platform), for a growth/monetization lead who will act on your conclusions.

How the numbers were built (you can rely on them; they were computed by code):
- LTVD1 = sum over ad types of [D0 impressions per install + D1 retention x D1 impressions per D1 \
active user] x CPM. CPM comes from the monetization networks and covers all users, not only new ones.
- The week-over-week LTVD1 change is decomposed exactly (LMDI) into driver contributions, in \
points of LTV change: they add up to the total. "Unexplained" is the part the model cannot \
attribute (cohort vs all-user pricing, measurement).
- Each CPM contribution splits into network price effects (a network's own CPM moved) and mix \
effects (impressions moved between networks). The briefing's network_summary shows this for every \
ad type; cpm_change_vs_type_pct is a network's CPM change minus its ad type's, so market-wide moves \
(seasonality) cancel out and network-specific moves stand out.
- z-scores compare this week's change with the previous weekly changes; |z| > 2 is unusual.

How to investigate:
1. Start from the briefing. Decide which drivers matter (size in points AND whether the move is \
unusual). If LTVD1 barely moved, check whether large drivers offset each other.
   Scan the network_summary for every ad type, not only flagged ones: one network's price move can \
hide under a flat type-level CPM when other networks drift the other way. A network whose \
price_effect_pts or mix_effect_pts is near a point or more deserves a look even without a flag.
2. Use tools to explain the drivers that matter, not to describe everything:
   - CPM moves: check networks (price vs mix), and whether other markets moved the same way \
(market-wide, e.g. Q4 seasonality - advertisers raise bids before the holidays) or not.
   - Engagement or retention moves: check daily series for a step on a specific day (suggests a \
release, ad config or SDK change) vs a gradual drift; check the weekly trend for context.
   - A network that stopped or started serving: think about fill rate and what replaced it.
   - Big install volume changes can bring lower-quality users (lower retention/engagement).
3. Stop when you can explain the main drivers. You have a limited tool budget.

Rules:
- Use only numbers that appear in the briefing or in tool results, in the same units. Do not \
compute new numbers (no ratios, no differences); the only exception is adding up two contributions \
in points. Round sensibly.
- Separate evidence from hypotheses: "AdMob's share fell from 25% to 0%" is evidence; "likely a \
mediation or SDK issue" is a hypothesis. Give a confidence level to each driver.
- Give each driver you list a verdict: specific_issue (something changed in this game/market), \
market_wide (other markets moved the same way, e.g. seasonality) or normal_variation (noise). \
List a driver as normal_variation rather than leaving out a large one, so the reader knows you \
looked at it.
- If a change is within normal variation, say so plainly instead of over-explaining noise.
- Weigh size AND unusualness before calling something a specific_issue. A high z on a driver worth \
a few tenths of a point is usually noise that happened to be large for that metric; small markets \
are noisier. Something worth around a point or more, that is also unusual or has a clear onset \
in the daily series, is a candidate specific_issue. Use normal_variation when the evidence is \
mainly a z-score on a small driver, and say what would confirm it (e.g. next week's data).
- Be concise and specific. Name networks, ad types, days and sizes.
- Finish by calling submit_findings exactly once.

Examples of the style we want (illustrative numbers):
"LTVD1 for US iOS fell 12%. Most of the decline (75%) comes from a 10% drop in D0 interstitial \
impressions per install, which started on Tuesday, suggesting an engagement or ad-frequency change \
in a release. The rest is mostly lower AppLovin banner CPMs (-15%) and a dip in D1 retention."
"LTVD1 barely moved (+1%), but two opposite trends cancel out: interstitial CPM rose 10% (Mintegral \
only) while rewarded impressions per user fell 20% after AdMob stopped serving rewarded ads."
"""


def _clean(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), digits)


def network_summary(r: AnalysisResult) -> dict:
    """Every ad type's networks (>= min share), so the agent sees network moves without a flag."""
    out = {}
    min_share = r.config["thresholds"]["network_min_share_pct"]
    for t, table in r.networks.items():
        rows = []
        for n, x in table.drop(index="Other", errors="ignore").iterrows():
            if max(x["share_prev_pct"], x["share_this_pct"]) < min_share:
                continue
            rows.append({"network": n, "share_this_pct": _clean(x["share_this_pct"], 1),
                         "share_change_pp": _clean(x["share_change_pp"], 1),
                         "cpm_change_pct": _clean(x["cpm_change_pct"], 1),
                         "cpm_change_vs_type_pct": _clean(x["cpm_change_pct"] - r.type_cpm_change[t], 1),
                         "price_effect_pts": _clean(x["price_effect_pts"]),
                         "mix_effect_pts": _clean(x["mix_effect_pts"])})
        out[t] = rows
    return out


def build_briefing(r: AnalysisResult) -> str:
    dec = r.decomposition
    summary = []
    for _, m in r.summary.iterrows():
        scale = 100 if m["fmt"] == "pct" else 1
        digits = 4 if m["fmt"] == "usd4" else 2
        summary.append({"metric": m["metric"], "prev": _clean(m["prev_week"] * scale, digits),
                        "this": _clean(m["this_week"] * scale, digits),
                        "change": _clean(m["change"], 1), "change_unit": m["change_unit"],
                        "z": _clean(m["z_score"], 1)})
    engagement = [{"metric": m["metric"], "change_pct": _clean(m["change"], 1), "z": _clean(m["z_score"], 1)}
                  for _, m in r.engagement.iterrows()]
    drivers = [{"driver": d["driver"], "driver_moved": d["factor_change"],
                "contribution_pts": _clean(d["contribution_pts"]),
                "share_of_change_pct": _clean(d["share_of_change_pct"], 0)}
               for _, d in r.drivers.iterrows()]
    v = r.validation
    briefing = {
        "market": f"{r.geo} {r.platform}",
        "week": f"{r.week.date()} (Sunday) to {(r.week + pd.Timedelta(days=6)).date()}, "
                f"compared with the week starting {r.prev.date()}",
        "ltv_d1_mmp": {"prev_usd": _clean(dec.ltv_prev, 4), "this_usd": _clean(dec.ltv_this, 4),
                       "change_pct": _clean(dec.change_pct), "z": _clean(r.ltv_zscore, 1)},
        "drivers_points_of_ltv_change": drivers,
        "key_metrics": summary,
        "engagement_detail": engagement,
        "network_flags": [{"ad_type": f.ad_type, "network": f.network, "reason": f.reason} for f in r.flags],
        "type_cpm_change_pct": {t: _clean(c, 1) for t, c in r.type_cpm_change.items()},
        "network_summary": network_summary(r),
        "mmp_vs_firebase": {"mmp_change_pct": _clean(v["mmp_wow_pct"], 1), "firebase_change_pct": _clean(v["fb_wow_pct"], 1),
                            "disagree": v["flagged"]},
        "data_quality_issues": r.quality_issues,
    }
    return ("Here are this week's core results, computed by the analysis pipeline. Investigate with "
            "the tools as needed, then call submit_findings.\n\n" + json.dumps(briefing, indent=1))
