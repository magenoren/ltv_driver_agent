"""Run the agent on every case, several times, and record what happened.

Runs happen in memory (no BigQuery): the evaluation measures the agent's judgment, and the data
layer is tested separately (including a BigQuery-vs-pandas equality test). Runs are independent
API calls, so a few run in parallel threads.
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd

from src.agent.loop import run_agent
from src.data.loader import _to_timestamps, from_raw
from src.eval.cases import Case
from src.eval.scoring import score_run
from src.pipeline import AnalysisResult, analyze
from src.report.html_report import write_report


def market_loader_from_raw(raw: dict) -> Callable:
    """All-market CPM data for compare_cpm_across_markets, from the in-memory world."""
    def load(week: pd.Timestamp, prev: pd.Timestamp) -> pd.DataFrame:
        mon = raw["monetization"].copy()
        mon["date_week"] = pd.to_datetime(mon["date_week"].astype(str))
        mon = mon[mon["date_week"].isin([prev, week])]
        df = mon.groupby(["date_week", "country", "platform", "type"], as_index=False)[["impressions", "revenue"]].sum()
        return _to_timestamps(df, ["date_week"])
    return load


def prepare(case: Case, config: dict) -> AnalysisResult:
    n = config["baseline_weeks"]
    data = from_raw(case.raw, case.geo, case.platform, case.week - pd.Timedelta(weeks=n + 1), case.week)
    r = analyze(data, case.geo, case.platform, case.week, "memory", config)
    r.market_loader = market_loader_from_raw(case.raw)
    return r


def _one_run(case: Case, r: AnalysisResult, run: int, client_factory, reports_dir: Path | None) -> dict:
    record = {"case": case.id, "run": run, "error": None, "findings": None, "tool_calls": 0,
              "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.0,
              "uncached_cost_usd": 0.0, "unverified": 0, "malformed": 0,
              "unknown_tools": 0, "tools_used": [], "report": None}
    out = None
    try:
        out = run_agent(r, client=client_factory())
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
    if out is not None:
        record.update(findings=out.findings, tool_calls=out.tool_calls, input_tokens=out.input_tokens,
                      output_tokens=out.output_tokens, cache_read_tokens=out.usage.cache_read,
                      cost_usd=out.cost_usd, uncached_cost_usd=out.uncached_cost_usd, unverified=len(out.warnings),
                      malformed=sum(1 for s in out.trace if "structure_problems" in s.output),
                      unknown_tools=sum(1 for s in out.trace if s.kind == "tool" and "Unknown tool" in str(s.output)),
                      tools_used=[s.tool for s in out.trace if s.kind == "tool"])
    score = score_run(case, record["findings"])
    record.update(found=score.found, network_named=score.network_named,
                  verdict_given=score.verdict_given, false_alarms=score.false_alarms,
                  false_alarm_confidence=score.false_alarm_confidence,
                  artifact_recognized=score.artifact_recognized)
    if reports_dir is not None:
        path = write_report(r, out, record["error"], out_dir=reports_dir, filename=f"{case.id}_run{run}.html")
        record["report"] = path.name
    return record


def run_eval(cases: list[Case], runs: int, config: dict, client_factory: Callable,
             workers: int = 3, reports_dir: Path | None = None,
             progress: Callable[[str], None] = print) -> list[dict]:
    prepared = {c.id: prepare(c, config) for c in cases}   # deterministic: compute once per case
    jobs = [(c, run) for c in cases for run in range(1, runs + 1)]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one_run, c, prepared[c.id], run, client_factory, reports_dir): (c, run)
                   for c, run in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            found = sum(rec["found"].values())
            status = rec["error"] or (f"found {found}/{len(rec['found'])}, "
                                      f"false alarms {len(rec['false_alarms'])}, {rec['tool_calls']} tools")
            progress(f"[{i}/{len(jobs)}] {rec['case']} run {rec['run']}: {status}")
            records.append(rec)
    return sorted(records, key=lambda x: (x["case"], x["run"]))
