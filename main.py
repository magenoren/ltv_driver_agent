"""Weekly LTVD1 driver analysis for one geo and platform.

    python main.py --geo US --platform ios
    python main.py --geo GB --platform android --week 2026-11-01 --source csv --open
    python main.py --geo US --platform ios --no-agent       # skip the AI, rule-based summary only

Runs the analysis pipeline, lets the Claude agent investigate and explain the drivers,
prints the result and writes an HTML report to outputs/.
"""
from __future__ import annotations

import argparse
import os
import traceback
import webbrowser

from src.pipeline import run
from src.report import format as fmt
from src.report.html_report import rule_based_summary, write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain week-over-week LTVD1 changes.")
    p.add_argument("--geo", required=True, help="country code, e.g. US")
    p.add_argument("--platform", required=True, choices=["ios", "android"])
    p.add_argument("--week", help="week start (a Sunday, YYYY-MM-DD); default: latest complete week")
    p.add_argument("--source", default="bq", choices=["bq", "csv"],
                   help="bq = BigQuery (default), csv = local files from generate_data.py")
    p.add_argument("--no-agent", action="store_true", help="skip the AI agent (rule-based summary only)")
    p.add_argument("--open", action="store_true", help="open the report in your browser")
    p.add_argument("--debug", action="store_true", help="print full tracebacks if the agent fails")
    return p.parse_args()


def run_agent_safely(r, debug: bool = False):
    """The report must still be produced if the agent can't run (no key, network, API error)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY is not set in .env"
    from src.agent.loop import run_agent

    print("\nAgent investigating", end="", flush=True)
    try:
        out = run_agent(r)
    except Exception as e:
        print()
        if debug:
            traceback.print_exc()
        return None, f"{type(e).__name__}: {e} (rerun with --debug for the full traceback)"
    for step in out.trace:
        if step.kind == "tool":
            print(f"\n  - {step.tool}({', '.join(f'{k}={v}' for k, v in step.inputs.items())})", end="")
    print()
    return out, None


def main() -> None:
    args = parse_args()
    try:
        r = run(args.geo, args.platform, args.week, args.source)
    except ValueError as e:          # bad --week, no complete week, ...
        raise SystemExit(str(e))

    print(f"{r.geo} {r.platform} | week {r.week.date()} vs {r.prev.date()} | source: {r.source}")
    for issue in r.quality_issues:
        print(f"  ! {issue}")

    print("\nDrivers (points of LTV change):")
    for _, d in r.drivers.iterrows():
        if abs(d["contribution_pts"]) >= 0.1:
            print(f"  {fmt.signed(d['contribution_pts'], digits=2):>7}  {d['driver']} ({fmt.minus(d['factor_change'])})")

    agent, agent_error = (None, None) if args.no_agent else run_agent_safely(r, args.debug)
    print()
    if agent and agent.findings:
        f = agent.findings
        print(f"{f['headline']}\n\n{f['summary']}")
        for s in f.get("next_steps", []):
            print(f"  next: {s}")
        if agent.warnings:
            print(f"\n  ! {len(agent.warnings)} unverified number(s) - see the report")
        u = agent.usage
        print(f"\n({agent.tool_calls} tool calls · input {u.total_input:,} tokens, of which {u.cache_read:,} read from "
              f"cache · output {u.output:,} · est. ${agent.cost_usd:.3f} vs ${agent.uncached_cost_usd:.3f} uncached)")
    else:
        if agent_error:
            print(f"Agent not used: {agent_error}\n")
        for sentence in rule_based_summary(r):
            print(f"- {sentence}")

    path = write_report(r, agent, agent_error)
    print(f"\nReport: {path}")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
