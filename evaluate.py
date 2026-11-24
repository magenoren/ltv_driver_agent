"""Evaluate the agent against planted ground truth.

    python evaluate.py --set dev --runs 3              # hand-written world (for tuning)
    python evaluate.py --set test --runs 3             # random held-out worlds (honest numbers)
    python evaluate.py --set stress --runs 3           # hand-designed hard cases (stress.yaml)
    python evaluate.py --set dev --runs 1 --markets US-ios,GB-android
    python evaluate.py --set test --seed 2028          # a fresh held-out world (after tuning on the others)
    python evaluate.py --set dev --plan                 # show the cases and true effect sizes only

Writes outputs/eval/<set>_<timestamp>/scorecard.html, results.json and one report per run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import webbrowser

from src.eval.cases import dev_cases, holdout_cases, stress_cases
from src.settings import PROJECT_ROOT, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the LTV driver agent on planted scenarios.")
    p.add_argument("--set", choices=["dev", "test", "stress"], default="dev")
    p.add_argument("--runs", type=int, default=3, help="runs per market (the model is not deterministic)")
    p.add_argument("--markets", help="comma-separated, e.g. US-ios,GB-android (default: all)")
    p.add_argument("--workers", type=int, default=3, help="runs in parallel")
    p.add_argument("--seed", type=int, help="test set only: use this seed instead of eval.test_seed in config.yaml")
    p.add_argument("--plan", action="store_true", help="only list cases and true effect sizes; no API calls")
    p.add_argument("--no-reports", action="store_true", help="don't save an HTML report per run")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    p.add_argument("--open", action="store_true", help="open the scorecard when done")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config, synthetic = load_yaml("config.yaml"), load_yaml("synthetic.yaml")
    markets = None
    if args.markets:
        markets = {(m.split("-")[0].upper(), m.split("-")[1].lower()) for m in args.markets.split(",")}
    if args.seed is not None:
        config["eval"]["test_seed"] = args.seed
    if args.set == "stress":
        cases = stress_cases(synthetic, load_yaml("stress.yaml"), config["eval"], config["baseline_weeks"], markets)
    else:
        build = dev_cases if args.set == "dev" else holdout_cases
        cases = build(synthetic, config["eval"], config["baseline_weeks"], markets)

    print(f"{args.set} set: {len(cases)} markets")
    if args.set != "test" or args.plan:   # never print the test set's answers unless explicitly asked
        for c in cases:
            planted = "; ".join(
                f"{s.description} (" + ("not scored" if s.effect_pts is None else
                                        f"true {s.effect_pts:+.2f}, observed {s.observed_pts:+.2f} pts, "
                                        + ("minor" if not s.material else "material" if s.detectable
                                           else "material but UNDETECTABLE")) + ")"
                for s in c.scenarios)
            planted = "; ".join(p for p in [planted] + [f"DATA ISSUE: {d.get('description', d['kind'])}"
                                                        for d in c.data_issues] if p) or "control"
            print(f"  {c.geo} {c.platform}: {planted}")
    if args.plan:
        return

    n_runs = len(cases) * args.runs
    est = n_runs * config["eval"]["est_input_tokens_per_run"]
    print(f"\n{n_runs} agent runs with {config['agent']['model']}, roughly {est / 1e6:.1f}M input tokens.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in .env")
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return

    import anthropic
    from src.eval.runner import run_eval
    from src.eval.scorecard import summarize, write_scorecard

    out_dir = PROJECT_ROOT / "outputs" / "eval" / f"{args.set}_{dt.datetime.now():%Y%m%d_%H%M}"
    records = run_eval(cases, args.runs, config, client_factory=anthropic.Anthropic, workers=args.workers,
                       reports_dir=None if args.no_reports else out_dir / "reports")
    path = write_scorecard(out_dir, args.set, cases, records, config, args.runs)
    s = summarize(cases, records)
    fmt = lambda v, d=0: "–" if v is None else f"{v:.{d}f}"
    print(f"\nDetectable causes found: {fmt(s['recall_detectable_pct'])}% ({s['recall_detectable']})"
          f" | all material: {fmt(s['recall_material_pct'])}% ({s['recall_material']})"
          f" | control runs clean: {fmt(s['control_clean_pct'])}%"
          f" | false alarms/run: {fmt(s['false_alarms_per_run'], 2)} ({fmt(s['high_conf_false_alarms_per_run'], 2)} high-confidence)")
    if s["artifact_runs"]:
        print(f"Data issues recognized: {fmt(s['artifact_recognized_pct'])}% of {s['artifact_runs']} runs")
    print(f"Runs with unverified numbers: {fmt(s['unverified_runs_pct'])}% | errors: {s['errors']}")
    print(f"Cost: ${s['total_cost_usd']:.2f} (${s['total_uncached_cost_usd']:.2f} without caching), "
          f"{fmt(s['cache_read_share_pct'])}% of input tokens read from cache")
    print(f"Scorecard: {path}")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
