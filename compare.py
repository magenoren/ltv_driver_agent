"""Compare two evaluations side by side (e.g. before and after a prompt change).

    python compare.py outputs/eval/stress_20261118_1400 outputs/eval/stress_20261119_1015

Metrics are recomputed from each run's raw records with the CURRENT scoring code, so an older
evaluation is compared on equal terms (e.g. it gets detectability even if it ran before it existed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.cases import dev_cases, holdout_cases, stress_cases
from src.eval.scorecard import summarize
from src.settings import load_yaml

METRICS = [
    ("recall_detectable_pct", "Detectable causes found", "%"),
    ("recall_material_pct", "All material causes found", "%"),
    ("control_clean_pct", "Control runs clean", "%"),
    ("false_alarms_per_run", "False alarms per run", ""),
    ("high_conf_false_alarms_per_run", "  of which high-confidence", ""),
    ("artifact_recognized_pct", "Data issues recognized", "%"),
    ("unverified_runs_pct", "Runs with unverified numbers", "%"),
    ("avg_tool_calls", "Tool calls per run", ""),
    ("cost_per_run_usd", "Cost per run, $", ""),
]


def load(path: Path):
    data = json.loads((path / "results.json").read_text(encoding="utf-8"))
    config, synthetic = load_yaml("config.yaml"), load_yaml("synthetic.yaml")
    if data.get("test_seed") is not None:
        config["eval"]["test_seed"] = data["test_seed"]
    n = config["baseline_weeks"]
    if data["set"] == "stress":
        cases = stress_cases(synthetic, load_yaml("stress.yaml"), config["eval"], n)
    elif data["set"] == "test":
        cases = holdout_cases(synthetic, config["eval"], n)
    else:
        cases = dev_cases(synthetic, config["eval"], n)
    ids = {r["case"] for r in data["records"]}
    cases = [c for c in cases if c.id in ids]
    return data, cases, summarize(cases, data["records"])


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two evaluation results.")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    args = p.parse_args()
    (d1, cases, s1), (d2, _, s2) = load(args.before), load(args.after)
    if d1["set"] != d2["set"]:
        print(f"Warning: comparing different sets ({d1['set']} vs {d2['set']})")

    print(f"\n{'':32s}{args.before.name:>26s}{args.after.name:>26s}")
    for key, label, unit in METRICS:
        a, b = s1.get(key), s2.get(key)
        fa = "–" if a is None else (f"{a:.0f}{unit}" if unit else f"{a:.2f}")
        fb = "–" if b is None else (f"{b:.0f}{unit}" if unit else f"{b:.2f}")
        print(f"{label:32s}{fa:>26s}{fb:>26s}")

    print("\nPer scenario (found / runs):")
    for c in cases:
        for s in c.scored:
            def rate(d):
                runs = [r for r in d["records"] if r["case"] == c.id and not r["error"]]
                return f"{sum(r['found'].get(s.id, False) for r in runs)}/{len(runs)}"
            tag = "" if s.detectable else " (minor)" if not s.material else " (undetectable)"
            print(f"  {c.geo} {c.platform:8s} {s.description[:50]:50s}{tag:15s}{rate(d1):>6s} -> {rate(d2)}")
        if c.is_control or c.data_issues:
            def fa(d):
                return sum(len(r["false_alarms"]) for r in d["records"] if r["case"] == c.id)
            label = "control" if c.is_control else "data issue"
            print(f"  {c.geo} {c.platform:8s} {label + ': false alarms':65s}{fa(d1):>6d} -> {fa(d2)}")


if __name__ == "__main__":
    main()
