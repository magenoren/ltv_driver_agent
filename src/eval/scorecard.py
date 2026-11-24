"""Aggregate run records into metrics, and write the scorecard (HTML + JSON)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from statistics import mean

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.eval.cases import Case


def _pct(num: float, den: float) -> float | None:
    return None if den == 0 else 100 * num / den


def summarize(cases: list[Case], records: list[dict]) -> dict:
    by_case = {c.id: c for c in cases}
    ok = [r for r in records if not r["error"]]

    def rate(include) -> tuple[int, int]:
        hits = total = 0
        for r in ok:
            for s in by_case[r["case"]].scored:
                if include(s):
                    total += 1
                    hits += r["found"].get(s.id, False)
        return hits, total

    det_hits, det_total = rate(lambda s: s.detectable)
    mat_hits, mat_total = rate(lambda s: s.material)
    all_hits, all_total = rate(lambda s: True)
    high_conf_fa = [sum(1 for c in r.get("false_alarm_confidence", []) if c == "high") for r in ok]
    net_named = [v for r in ok for sid, v in r["network_named"].items() if r["found"].get(sid)]
    control_runs = [r for r in ok if by_case[r["case"]].is_control]
    artifact_runs = [r for r in ok if r.get("artifact_recognized") is not None]
    return {
        "runs": len(records),
        "errors": len(records) - len(ok),
        "recall_detectable_pct": _pct(det_hits, det_total), "recall_detectable": f"{det_hits}/{det_total}",
        "recall_material_pct": _pct(mat_hits, mat_total), "recall_material": f"{mat_hits}/{mat_total}",
        "recall_all_pct": _pct(all_hits, all_total), "recall_all": f"{all_hits}/{all_total}",
        "network_named_pct": _pct(sum(net_named), len(net_named)), "network_named": f"{sum(net_named)}/{len(net_named)}",
        "control_runs": len(control_runs),
        "control_clean_pct": _pct(sum(1 for r in control_runs if not r["false_alarms"]), len(control_runs)),
        "false_alarms_per_run": mean(len(r["false_alarms"]) for r in ok) if ok else None,
        "high_conf_false_alarms_per_run": mean(high_conf_fa) if ok else None,
        "artifact_runs": len(artifact_runs),
        "artifact_recognized_pct": _pct(sum(1 for r in artifact_runs if r["artifact_recognized"]), len(artifact_runs)),
        "unverified_runs_pct": _pct(sum(1 for r in ok if r["unverified"]), len(ok)),
        "malformed_runs_pct": _pct(sum(1 for r in ok if r["malformed"]), len(ok)),
        "avg_tool_calls": mean(r["tool_calls"] for r in ok) if ok else None,
        "avg_input_tokens": mean(r["input_tokens"] for r in ok) if ok else None,
        "total_input_tokens": sum(r["input_tokens"] for r in records),
        "cache_read_share_pct": _pct(sum(r.get("cache_read_tokens", 0) for r in records),
                                     sum(r["input_tokens"] for r in records)),
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in records),
        "total_uncached_cost_usd": sum(r.get("uncached_cost_usd", 0) for r in records),
        "cost_per_run_usd": mean(r.get("cost_usd", 0) for r in ok) if ok else None,
        "total_output_tokens": sum(r["output_tokens"] for r in records),
    }


def scenario_rows(cases: list[Case], records: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        runs = [r for r in records if r["case"] == c.id and not r["error"]]
        for d in c.data_issues:
            rec = [r.get("artifact_recognized") for r in runs]
            rows.append({"market": f"{c.geo} {c.platform}", "world": c.world,
                         "scenario": f"Data issue: {d.get('description', d['kind'])}",
                         "effect": "", "observed": "", "material": "data issue",
                         "found": f"{sum(1 for x in rec if x)}/{len(rec)} recognized",
                         "marks": ["hit" if x else "miss" for x in rec], "verdicts": "", "network": "",
                         "false_alarms": sum(len(r["false_alarms"]) for r in runs) if not c.scenarios else "",
                         "control": False})
        if c.is_control:
            fa = [len(r["false_alarms"]) for r in runs]
            rows.append({"market": f"{c.geo} {c.platform}", "world": c.world, "scenario": "Control: nothing planted",
                         "effect": "", "observed": "", "material": "", "found": "", "marks": [],
                         "verdicts": "", "network": "",
                         "false_alarms": sum(fa), "control": True})
            continue
        for i, s in enumerate(c.scenarios):
            miss = "miss" if s.detectable else "miss-minor"   # missing a minor or undetectable effect is not a failure
            marks = [("hit" if r["found"].get(s.id) else miss) for r in runs] if s.expected_driver else []
            verdicts = sorted({str(r["verdict_given"].get(s.id) or "not listed") for r in runs}) if s.expected_driver else []
            named = [r["network_named"][s.id] for r in runs if s.id in r["network_named"] and r["found"].get(s.id)]
            rows.append({
                "market": f"{c.geo} {c.platform}", "world": c.world,
                "scenario": s.description or s.id,
                "effect": "" if s.effect_pts is None else f"{s.effect_pts:+.2f}".replace("-", "−"),
                "material": "" if not s.expected_driver else (
                    "minor" if not s.material else "material" if s.detectable else "undetectable"),
                "observed": "" if s.observed_pts is None else f"{s.observed_pts:+.2f}".replace("-", "−"),
                "found": "not scored (context)" if not s.expected_driver else f"{marks.count('hit')}/{len(marks)}",
                "marks": marks, "verdicts": ", ".join(v.replace("_", " ") for v in verdicts),
                "network": f"{sum(named)}/{len(named)}" if named else "",
                # false alarms belong to the market, not to a scenario: show them once
                "false_alarms": sum(len(r["false_alarms"]) for r in runs) if i == 0 else "", "control": False,
            })
    return rows


def write_scorecard(out_dir: Path, set_name: str, cases: list[Case], records: list[dict],
                    config: dict, runs: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(cases, records)
    (out_dir / "results.json").write_text(json.dumps({
        "set": set_name, "runs_per_case": runs, "model": config["agent"]["model"],
        "test_seed": config["eval"]["test_seed"], "summary": summary,
        "cases": [{"id": c.id, "geo": c.geo, "platform": c.platform, "world": c.world,
                   "scenarios": [{**s.__dict__, "start": str(s.start)} for s in c.scenarios]} for c in cases],
        "records": [{k: v for k, v in r.items() if k != "traceback"} for r in records],
    }, indent=1, default=str), encoding="utf-8")

    env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html", "j2"]))
    html = env.get_template("scorecard.html.j2").render(
        set_name=set_name, runs=runs, n_cases=len(cases), model=config["agent"]["model"],
        min_effect=config["eval"]["min_effect_pts"], s=summary, rows=scenario_rows(cases, records),
        records=records, generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    path = out_dir / "scorecard.html"
    path.write_text(html, encoding="utf-8")
    return path
