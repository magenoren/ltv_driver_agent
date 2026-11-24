"""Tests for the evaluation harness (no API calls)."""
import copy

import pandas as pd
import pytest

from src.eval.cases import dev_cases, holdout_cases, random_world
from src.eval.runner import run_eval
from src.eval.scorecard import summarize, write_scorecard
from src.eval.scoring import score_run
from src.settings import load_yaml
from tests.test_agent import FakeClient, tool_use

CONFIG = load_yaml("config.yaml")
SYNTH = load_yaml("synthetic.yaml")
N = CONFIG["baseline_weeks"]


@pytest.fixture(scope="module")
def dev():
    return {c.id: c for c in dev_cases(SYNTH, CONFIG["eval"], N)}


def entry(driver, verdict="specific_issue", network=None, explanation="x"):
    return {"driver": driver, "verdict": verdict, "network": network, "explanation": explanation, "confidence": "high"}


# --------------------------------------------------------------------------- cases
def test_dev_cases_have_true_effect_sizes(dev):
    us = dev["dev_w0_US_ios"]
    by_id = {s.id: s for s in us.scenarios}
    assert by_id["us_ios_inter_d0"].material and by_id["us_ios_inter_d0"].effect_pts < -5
    assert not by_id["us_ios_applovin_banner"].material          # small: a bonus, not a requirement
    assert dev["dev_w0_DE_ios"].is_control
    installs = next(s for s in dev["dev_w0_BR_android"].scenarios if s.target == "installs")
    assert installs.expected_driver is None                       # context, not a driver


def test_counterfactual_separates_planted_effect_from_seasonality(dev):
    mintegral = next(s for s in dev["dev_w0_GB_android"].scenarios if s.network == "Mintegral")
    assert 3.5 < mintegral.effect_pts < 5.5   # the CPM driver is ~+6 pts; the rest is the Q4 ramp


def test_random_worlds_are_reproducible_and_scoreable():
    a = random_world(SYNTH, seed=99, control_share=0.3)
    b = random_world(SYNTH, seed=99, control_share=0.3)
    assert a["scenarios"] == b["scenarios"]
    for geo, platform in {(s["geo"], s["platform"]) for s in a["scenarios"]}:
        drivers = [(s["target"], s.get("ad_type")) for s in a["scenarios"] if (s["geo"], s["platform"]) == (geo, platform)]
        assert len(drivers) <= 2


def test_test_set_differs_from_dev_set():
    t = holdout_cases(SYNTH, {**CONFIG["eval"], "test_worlds": 1}, N)
    assert len(t) == 8 and all(c.set_name == "test" for c in t)
    dev_ids = {s["id"] for s in SYNTH["scenarios"]}
    assert not dev_ids & {s.id for c in t for s in c.scenarios}


# --------------------------------------------------------------------------- scoring
def test_scoring_found_verdict_and_false_alarms(dev):
    gb = dev["dev_w0_GB_android"]
    findings = {"drivers": [
        entry("Interstitial CPM", network="Mintegral"),
        entry("Rewarded impressions per install, D0", explanation="Fewer rewarded ads after AdMob stopped serving."),
        entry("Rewarded impressions per D1 active, D1", verdict="normal_variation"),   # named, but called noise
        entry("Banner CPM"),                                                           # not planted -> false alarm
        entry("D1 retention", verdict="normal_variation"),                             # fine: not a claim
    ]}
    s = score_run(gb, findings)
    ids = {sc.target + "_" + str(sc.ad_type): sc.id for sc in gb.scenarios}
    assert s.found[ids["cpm_Interstitial"]] and s.network_named[ids["cpm_Interstitial"]]
    assert s.found[ids["imps_d0_Rewarded"]]
    assert s.found[ids["network_share_Rewarded"]]      # AdMob named on a Rewarded driver
    assert not s.found[ids["imps_d1_Rewarded"]] and s.verdict_given[ids["imps_d1_Rewarded"]] == "normal_variation"
    assert s.false_alarms == ["Banner CPM"]


def test_no_findings_scores_as_all_missed(dev):
    s = score_run(dev["dev_w0_US_ios"], None)
    assert not any(s.found.values()) and s.false_alarms == []


# --------------------------------------------------------------------------- end to end with a fake model
def test_run_eval_end_to_end(dev, tmp_path):
    cases = [dev["dev_w0_US_ios"], dev["dev_w0_DE_ios"]]
    findings_by_market = {
        "US": {"headline": "LTVD1 fell 8.2%.", "summary": "D0 interstitials.", "next_steps": [],
               "drivers": [entry("Interstitial impressions per install, D0")]},
        "DE": {"headline": "Normal week.", "summary": "Noise.", "next_steps": [],
               "drivers": [entry("D1 retention")]},        # a false alarm on a control
    }

    class Factory:
        def __call__(self):
            return self

        @property
        def messages(self):
            return self

        def create(self, **kw):
            market = "US" if '"market": "US' in kw["messages"][0]["content"] else "DE"
            return FakeClient([[tool_use(1, "submit_findings", findings_by_market[market])]]).create(**kw)

    records = run_eval(cases, runs=2, config=CONFIG, client_factory=Factory(), workers=2,
                       reports_dir=tmp_path / "reports", progress=lambda m: None)
    assert len(records) == 4 and not any(r["error"] for r in records)
    s = summarize(cases, records)
    assert s["recall_material"] == "2/4"           # D0 interstitials found twice, retention missed twice
    assert s["control_clean_pct"] == 0             # DE false alarm in both runs
    path = write_scorecard(tmp_path, "dev", cases, records, CONFIG, runs=2)
    html = path.read_text(encoding="utf-8")
    assert "Agent evaluation: dev set" in html and "Control: nothing planted" in html
    assert (tmp_path / "results.json").exists() and len(list((tmp_path / "reports").glob("*.html"))) == 4


# --------------------------------------------------------------------------- stress set
@pytest.fixture(scope="module")
def stress():
    from src.eval.cases import stress_cases
    return {(c.geo, c.platform): c for c in stress_cases(SYNTH, load_yaml("stress.yaml"), CONFIG["eval"], N)}


def test_stress_scenarios_are_all_material(stress):
    scored = [s for c in stress.values() for s in c.scored]
    assert len(scored) == 6 and all(s.material for s in scored)
    assert stress[("DE", "android")].is_control and stress[("BR", "ios")].is_control
    assert not stress[("GB", "android")].is_control            # it has a data issue


def test_data_issue_removes_d1_rows_and_is_detected(stress):
    from src.eval.runner import prepare
    case = stress[("GB", "android")]
    fb = case.raw["firebase"]
    gone = fb[(fb.country == "GB") & (fb.platform == "android") & (fb.age == 1)
              & (pd.to_datetime(fb.install_date.astype(str)) == pd.Timestamp("2026-11-07"))]
    assert gone.empty
    r = prepare(case, CONFIG)
    assert any("no D1 users" in q for q in r.quality_issues)
    assert r.validation["flagged"]                             # MMP still has the D1 revenue


def test_real_twin_of_the_artifact_has_clean_data(stress):
    from src.eval.runner import prepare
    r = prepare(stress[("US", "android")], CONFIG)
    assert r.quality_issues == [] and not r.validation["flagged"]


def test_artifact_scoring(stress):
    case = stress[("GB", "android")]
    good = {"headline": "Retention drop is a data artifact.", "summary": "D1 data has not matured for Saturday.",
            "drivers": [entry("D1 retention", verdict="normal_variation")], "caveats": []}
    bad = {"headline": "Retention collapsed.", "summary": "Players stopped returning.",
           "drivers": [entry("D1 retention")], "caveats": []}
    assert score_run(case, good).artifact_recognized and score_run(case, good).false_alarms == []
    s = score_run(case, bad)
    assert s.artifact_recognized is False and s.false_alarms == ["D1 retention"]


def test_unflagged_network_event_is_not_flagged(stress):
    from src.eval.runner import prepare
    assert prepare(stress[("BR", "android")], CONFIG).flags == []   # the agent must find it unprompted


# --------------------------------------------------------------------------- detectability
def test_detectability_separates_hidden_effects(stress):
    gb = stress[("GB", "ios")].scored[0]
    assert gb.material and not gb.detectable          # true -1.09, but noise left only ~-0.56 visible
    br = stress[("BR", "android")].scored[0]
    assert br.detectable and 0.75 <= br.observed_pts  # visible at the network level (Mintegral price effect)


def test_undetectable_misses_are_not_counted_against_recall(stress):
    case = stress[("GB", "ios")]
    sid = case.scored[0].id
    records = [{"case": case.id, "run": 1, "error": None, "found": {sid: False}, "network_named": {},
                "verdict_given": {}, "false_alarms": ["Interstitial impressions per install, D0"],
                "false_alarm_confidence": ["medium"], "unverified": 0, "malformed": 0, "tool_calls": 3,
                "input_tokens": 1, "output_tokens": 1}]
    s = summarize([case], records)
    assert s["recall_detectable"] == "0/0" and s["recall_material"] == "0/1"
    assert s["false_alarms_per_run"] == 1 and s["high_conf_false_alarms_per_run"] == 0


def test_false_alarm_confidence_is_recorded(dev):
    s = score_run(dev["dev_w0_DE_ios"], {"drivers": [entry("D1 retention")]})
    assert s.false_alarms == ["D1 retention"] and s.false_alarm_confidence == ["high"]


# --------------------------------------------------------------------------- agent change B
def test_briefing_shows_networks_for_every_type(stress):
    import json
    from src.agent.prompts import build_briefing
    from src.eval.runner import prepare
    b = build_briefing(prepare(stress[("BR", "android")], CONFIG))
    summary = json.loads(b[b.find("{"):])["network_summary"]
    assert set(summary) == {"Banner", "Interstitial", "Rewarded"}
    mintegral = next(n for n in summary["Interstitial"] if n["network"] == "Mintegral")
    assert mintegral["cpm_change_vs_type_pct"] > 5   # unflagged, but now visible


# --------------------------------------------------------------------------- compare.py
def test_compare_recomputes_both_sides(dev, tmp_path):
    import compare
    cases = [dev["dev_w0_US_ios"]]
    sid = cases[0].scored[0].id
    rec = lambda found: [{"case": cases[0].id, "run": 1, "error": None, "findings": None,
                          "found": {s.id: (s.id == sid and found) for s in cases[0].scored},
                          "network_named": {}, "verdict_given": {}, "false_alarms": [], "unverified": 0,
                          "malformed": 0, "tool_calls": 2, "input_tokens": 10, "output_tokens": 5}]
    for name, found in (("before", False), ("after", True)):
        write_scorecard(tmp_path / name, "dev", cases, rec(found), CONFIG, runs=1)
    _, _, before = compare.load(tmp_path / "before")
    _, _, after = compare.load(tmp_path / "after")
    assert before["recall_detectable_pct"] < after["recall_detectable_pct"]
