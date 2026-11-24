"""Turn an AnalysisResult into a single self-contained HTML file.

All formatting decisions are made here in Python; the Jinja2 template only places
ready-made strings. That keeps the template simple and the logic testable.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.data.loader import AD_TYPES
from src.pipeline import AnalysisResult
from src.report import charts
from src.report import format as fmt
from src.settings import PROJECT_ROOT

TEMPLATE_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BRIDGE_ROWS = 6  # biggest drivers shown in the bridge; the rest are grouped


# --------------------------------------------------------------------------- rule-based summary
def rule_based_summary(r: AnalysisResult) -> list[str]:
    """A plain, deterministic summary. It is the baseline the agent (step 5) has to beat."""
    dec, d = r.decomposition, r.drivers
    z = r.ltv_zscore
    unusual = ("an unusual move" if abs(z) > 2 else "within normal weekly variation") if not math.isnan(z) \
        else "too little history to judge"
    s = [f"LTVD1 (MMP) {fmt.movement(dec.change_pct)} week over week, "
         f"from ${dec.ltv_prev:.4f} to ${dec.ltv_this:.4f} per install: {unusual}"
         + (f" (z = {fmt.signed(z)})." if not math.isnan(z) else ".")]

    notable = d[d["contribution_pts"].abs() >= 0.5]
    if d["share_of_change_pct"].notna().any():   # a real change, mostly in one direction
        top = d.iloc[0]
        s.append(f"The largest driver is {top['driver']} ({fmt.minus(top['factor_change'])}), worth "
                 f"{fmt.signed(top['contribution_pts'], digits=2)} points, about "
                 f"{top['share_of_change_pct']:.0f}% of the change.")
        others = notable.iloc[1:4]
        if not others.empty:
            s.append("Other drivers above half a point: " + "; ".join(
                f"{x['driver']} {fmt.minus(x['factor_change'])} ({fmt.signed(x['contribution_pts'], digits=2)} pts)"
                for _, x in others.iterrows()) + ".")
    else:
        # Only call it "offsetting" when both sides are sizeable; smaller splits are ordinary noise
        pos = notable[notable["contribution_pts"] >= 1]
        neg = notable[notable["contribution_pts"] <= -1]
        if not pos.empty and not neg.empty:
            p, n = pos.iloc[0], neg.iloc[0]
            s.append(f"The total moved little, but drivers offset each other: {p['driver']} "
                     f"({fmt.minus(p['factor_change'])}, {fmt.signed(p['contribution_pts'], digits=2)} pts) against "
                     f"{n['driver']} ({fmt.minus(n['factor_change'])}, {fmt.signed(n['contribution_pts'], digits=2)} pts).")
        elif notable.empty:
            s.append("No single driver moved LTV by more than half a point.")
        else:
            x = notable.iloc[0]
            s.append(f"The main driver is {x['driver']} ({fmt.minus(x['factor_change'])}, "
                     f"{fmt.signed(x['contribution_pts'], digits=2)} pts).")

    if r.flags:
        by_type = []
        for t in AD_TYPES:
            reasons = [f.reason for f in r.flags_for(t)]
            if reasons:
                by_type.append(f"{t} ({', '.join(reasons[:3])}{', …' if len(reasons) > 3 else ''})")
        s.append("Network changes flagged: " + "; ".join(by_type) + ".")
    if r.validation["flagged"]:
        s.append(f"MMP and Firebase disagree on the change ({fmt.signed(r.validation['mmp_wow_pct'], '%')} vs "
                 f"{fmt.signed(r.validation['fb_wow_pct'], '%')}); figures use MMP.")
    resid = d.loc[d["category"] == "Unexplained", "contribution_pts"].sum()
    if abs(resid) > 1:
        s.append(f"{fmt.signed(resid, digits=2)} points of the change could not be attributed to a driver.")
    if r.quality_issues:
        s.append("Data checks found issues; see the data checks section before relying on these numbers.")
    return s


# --------------------------------------------------------------------------- context
def day(ts, year: bool = False) -> str:
    """'Nov 1' or 'Nov 1, 2026'. (strftime's %-d drops the leading zero on Linux/macOS
    but crashes on Windows, so we build it by hand.)"""
    return f"{ts:%b} {ts.day}" + (f", {ts.year}" if year else "")


def _metric_rows(table, weekly, week, n_weeks):
    rows = []
    for _, m in table.iterrows():
        history = weekly.loc[:week, m["column"]].tail(n_weeks + 1).tolist()
        z = m["z_score"]
        rows.append({
            "metric": m["metric"],
            "prev": fmt.level(m["prev_week"], m["fmt"]),
            "this": fmt.level(m["this_week"], m["fmt"]),
            "change": fmt.signed(m["change"], m["change_unit"]),
            "cls": fmt.direction(m["change"], 0.05),
            "z": "" if math.isnan(z) else fmt.signed(z),
            "unusual": (not math.isnan(z)) and abs(z) > 2,
            "spark": charts.sparkline(history),
        })
    return rows


def _network_sections(r: AnalysisResult):
    sections = []
    for t in AD_TYPES:
        flags = r.flags_for(t)
        flagged_networks = {f.network for f in flags if f.network}
        rows = []
        for network, x in r.networks[t].iterrows():
            rows.append({
                "network": network,
                "share_prev": f"{x['share_prev_pct']:.1f}%",
                "share_this": f"{x['share_this_pct']:.1f}%",
                "share_change": fmt.signed(x["share_change_pp"], "pp"),
                "cpm_prev": fmt.level(x["cpm_prev"], "usd"),
                "cpm_this": fmt.level(x["cpm_this"], "usd"),
                "cpm_change": fmt.signed(x["cpm_change_pct"], "%"),
                "cpm_cls": fmt.direction(x["cpm_change_pct"], 0.05),
                "price": fmt.signed(x["price_effect_pts"], digits=2),
                "price_cls": fmt.direction(x["price_effect_pts"], 0.005),
                "mix": fmt.signed(x["mix_effect_pts"], digits=2),
                "mix_cls": fmt.direction(x["mix_effect_pts"], 0.005),
                "flagged": network in flagged_networks,
            })
        sections.append({"type": t, "cpm_change": fmt.signed(r.type_cpm_change[t], "%"),
                         "cpm_cls": fmt.direction(r.type_cpm_change[t], 0.05),
                         "flagged": bool(flags), "reasons": [f.reason for f in flags], "rows": rows})
    return sections


def _agent_context(agent, agent_error: str | None) -> dict | None:
    """Agent findings and trace, ready for the template. None -> the rule-based summary is shown."""
    if agent is None or agent.findings is None:
        return None
    f = agent.findings
    steps = []
    for st in agent.trace:
        if st.kind == "tool":
            args = ", ".join(f"{k}={v}" for k, v in st.inputs.items())
            steps.append({"title": f"{st.tool}({args})", "reasoning": st.reasoning,
                          "output": json.dumps(st.output, indent=1)[:4000]})
        elif st.kind == "validation" and "structure_problems" in st.output:
            steps.append({"title": "submit_findings (malformed, sent back)", "reasoning": st.reasoning,
                          "output": "\n".join(st.output["structure_problems"])})
        elif st.kind == "validation":
            n = len(st.output.get("unverified_numbers", []))
            steps.append({"title": "submit_findings", "reasoning": st.reasoning,
                          "output": ("All numbers verified against the briefing and tool results." if n == 0 else
                                     f"{n} unverified number(s):\n" + "\n".join(st.output["unverified_numbers"]))})
        elif st.reasoning:
            steps.append({"title": "note", "reasoning": st.reasoning, "output": ""})
    return {
        "headline": f.get("headline", ""), "summary": f.get("summary", ""),
        "drivers": f.get("drivers", []), "next_steps": f.get("next_steps", []), "caveats": f.get("caveats", []),
        "steps": steps, "warnings": agent.warnings, "model": agent.model, "tool_calls": agent.tool_calls,
        "tokens": (f"{agent.usage.total_input:,} input tokens ({agent.usage.cache_read:,} from cache), "
                   f"{agent.usage.output:,} output tokens, est. ${agent.cost_usd:.3f}"),
    }


def build_context(r: AnalysisResult, agent=None, agent_error: str | None = None) -> dict:
    dec, n_weeks = r.decomposition, r.config["baseline_weeks"]
    week_end = r.week + dt.timedelta(days=6)

    d = r.drivers
    shown = d.head(BRIDGE_ROWS)
    bridge_rows = [(x["driver"], x["contribution_pts"]) for _, x in shown.iterrows()]
    rest = d.iloc[BRIDGE_ROWS:]
    if not rest.empty:
        bridge_rows.append((f"{len(rest)} smaller drivers", rest["contribution_pts"].sum()))

    driver_rows = [{
        "driver": x["driver"], "category": x["category"], "factor": fmt.minus(x["factor_change"]),
        "pts": fmt.signed(x["contribution_pts"], digits=2), "cls": fmt.direction(x["contribution_pts"], 0.005),
        "share": "" if math.isnan(x["share_of_change_pct"]) else f"{x['share_of_change_pct']:.0f}%",
    } for _, x in d.iterrows()]

    v = r.validation
    resid = d.loc[d["category"] == "Unexplained", "contribution_pts"].sum()
    th = r.config["thresholds"]
    return {
        "title": f"LTVD1 drivers: {r.geo} {r.platform}, week of {day(r.week, year=True)}",
        "geo": r.geo, "platform": "iOS" if r.platform == "ios" else "Android",
        "period": f"{day(r.week)} to {day(week_end, year=True)}",
        "prev_period": f"week of {day(r.prev)}",
        "headline": f"LTVD1 {fmt.movement(dec.change_pct)}",
        "headline_cls": fmt.direction(dec.change_pct, 0.5),
        "ltv_prev": f"${dec.ltv_prev:.4f}", "ltv_this": f"${dec.ltv_this:.4f}",
        "ltv_z": "" if math.isnan(r.ltv_zscore) else fmt.signed(r.ltv_zscore),
        "ltv_unusual": (not math.isnan(r.ltv_zscore)) and abs(r.ltv_zscore) > 2,
        "narrative": rule_based_summary(r),
        "agent": _agent_context(agent, agent_error),
        "agent_error": agent_error,
        "bridge_svg": charts.bridge(bridge_rows, dec.change_pct),
        "show_share": not d["share_of_change_pct"].isna().all(),
        "drivers": driver_rows,
        "kpis": _metric_rows(r.summary, r.weekly, r.week, n_weeks),
        "engagement": _metric_rows(r.engagement, r.weekly, r.week, n_weeks),
        "networks": _network_sections(r),
        "n_weeks": n_weeks,
        "quality_issues": r.quality_issues,
        "validation": {
            "mmp": fmt.signed(v["mmp_wow_pct"], "%"), "fb": fmt.signed(v["fb_wow_pct"], "%"),
            "diff": fmt.signed(v["wow_diff_pp"], "pp"), "gap": fmt.signed(v["level_gap_this_pct"], "%"),
            "flagged": v["flagged"], "threshold": v["threshold_pp"],
        },
        "residual": fmt.signed(resid, digits=2),
        "residual_large": abs(resid) > 1,
        "thresholds": th,
        "source": "BigQuery" if r.source == "bq" else "local CSV files",
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def render(context: dict) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html", "j2"]))
    return env.get_template("template.html.j2").render(**context)


def write_report(r: AnalysisResult, agent=None, agent_error: str | None = None,
                 out_dir: Path = OUTPUT_DIR, filename: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (filename or f"ltv_report_{r.geo}_{r.platform}_{r.week:%Y-%m-%d}.html")
    path.write_text(render(build_context(r, agent, agent_error)), encoding="utf-8")
    return path
