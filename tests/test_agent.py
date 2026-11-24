"""Agent tests - no API calls.

FakeClient replays a scripted conversation in the same shape the Anthropic SDK returns, so we
can test our side of the loop (tool execution, budget, validation, report) deterministically.
The real model is evaluated separately on the planted scenarios (step 6).
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.agent import tools
from src.agent.loop import run_agent
from src.agent.prompts import build_briefing
from src.agent.validation import NumberBook, check_findings, claims_in
from src.data.loader import _to_timestamps, from_raw
from src.data.synthetic import generate
from src.pipeline import analyze
from src.report.html_report import build_context, render
from src.settings import load_yaml

WEEK = pd.Timestamp("2026-11-01")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def raw():
    return generate(load_yaml("synthetic.yaml"))


@pytest.fixture(autouse=True)
def market_data_from_raw(raw, monkeypatch):
    """compare_cpm_across_markets normally queries BigQuery/CSV; feed it the in-memory world."""
    def fake_load_market_cpm(source, week, prev):
        mon = raw["monetization"].copy()
        mon["date_week"] = pd.to_datetime(mon["date_week"].astype(str))
        mon = mon[mon["date_week"].isin([prev, week])]
        df = mon.groupby(["date_week", "country", "platform", "type"], as_index=False)[["impressions", "revenue"]].sum()
        return _to_timestamps(df, ["date_week"])
    monkeypatch.setattr(tools, "load_market_cpm", fake_load_market_cpm)


def result(raw, geo, platform, **agent_cfg):
    cfg = load_yaml("config.yaml")
    cfg["agent"].update(agent_cfg)
    data = from_raw(raw, geo, platform, WEEK - pd.Timedelta(weeks=cfg["baseline_weeks"] + 1), WEEK)
    return analyze(data, geo, platform, WEEK, "csv", cfg)


# --------------------------------------------------------------------------- fake Claude
def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(i, name, inputs):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=inputs)


class FakeClient:
    def __init__(self, script, usage=None):
        self.script, self.requests, self.kwargs = list(script), [], []
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 200}
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        self.kwargs.append({k: v for k, v in kwargs.items() if k != "messages"})
        content = self.script.pop(0)
        return SimpleNamespace(content=content, usage=SimpleNamespace(**self.usage))


GOOD_FINDINGS = {
    "headline": "LTVD1 fell 8.2%, mostly because D0 interstitial impressions per install dropped 18.1%.",
    "summary": "The drop is worth -6.68 points, about 81% of the change. D1 retention fell 1.4pp (-1.19 pts).",
    "drivers": [{"driver": "Interstitial impressions per install, D0", "confidence": "high",
                 "verdict": "specific_issue",
                 "explanation": "Down 18.1% from the first day of the week, a step change."}],
    "next_steps": ["Check the release notes and interstitial frequency caps for the start of the week."],
}


# --------------------------------------------------------------------------- the loop
def test_agent_loop_runs_tools_rejects_invented_numbers_then_accepts(raw):
    r = result(raw, "US", "ios")
    bad = {**GOOD_FINDINGS, "summary": GOOD_FINDINGS["summary"] + " Playtime fell 23.7%."}
    client = FakeClient([
        [text("D0 interstitials dominate. Is it a step change?"),
         tool_use(1, "get_daily_series", {"metric": "imps_d0_per_install", "ad_type": "Interstitial"})],
        [text("Is the banner CPM move market-wide?"),
         tool_use(2, "compare_cpm_across_markets", {"ad_type": "Banner"})],
        [tool_use(3, "submit_findings", bad)],          # 23.7% was never given -> rejected
        [tool_use(4, "submit_findings", GOOD_FINDINGS)],
    ])
    out = run_agent(r, client=client)

    assert out.findings == GOOD_FINDINGS and out.warnings == []
    assert out.tool_calls == 2
    assert [s.kind for s in out.trace] == ["tool", "tool", "validation", "validation"]
    assert out.trace[0].reasoning.startswith("D0 interstitials")
    # the rejection was sent back to the model as an error tool_result
    rejection = client.requests[3][-1]["content"][0]
    assert rejection["is_error"] and "23.7%" in rejection["content"]
    # and the tool results it received were real data
    first_result = json.loads(client.requests[1][-1]["content"][0]["content"])
    assert first_result["metric"] == "imps_d0_per_install" and len(first_result["days"]) == 14


def test_tool_budget_is_enforced(raw):
    r = result(raw, "US", "ios", max_tool_calls=1)
    client = FakeClient([
        [tool_use(1, "get_weekly_trend", {"metric": "retention_d1"})],
        [tool_use(2, "get_weekly_trend", {"metric": "cpm_banner"})],   # over budget
        [tool_use(3, "submit_findings", GOOD_FINDINGS)],
    ])
    out = run_agent(r, client=client)
    assert out.tool_calls == 1
    assert "budget" in out.trace[1].output["error"]
    assert out.findings == GOOD_FINDINGS


def test_findings_accepted_with_warnings_after_retry_limit(raw):
    r = result(raw, "US", "ios", max_validation_retries=0)
    bad = {**GOOD_FINDINGS, "headline": "LTVD1 fell 9.9%."}
    out = run_agent(r, client=FakeClient([[tool_use(1, "submit_findings", bad)]]))
    assert out.findings == bad and len(out.warnings) == 1 and "9.9%" in out.warnings[0]


def test_driver_names_are_constrained_to_this_weeks_drivers(raw):
    r = result(raw, "US", "ios")
    schema = tools.submit_findings_schema(r.drivers["driver"].tolist())
    enum = schema["input_schema"]["properties"]["drivers"]["items"]["properties"]["driver"]["enum"]
    assert "D1 retention" in enum and len(enum) == len(r.drivers)


def test_report_shows_agent_findings_and_trace(raw):
    r = result(raw, "US", "ios")
    out = run_agent(r, client=FakeClient([
        [text("Check the daily pattern."), tool_use(1, "get_daily_series", {"metric": "retention_d1"})],
        [tool_use(2, "submit_findings", GOOD_FINDINGS)]]))
    html = render(build_context(r, out))
    assert GOOD_FINDINGS["headline"] in html
    assert "How the agent investigated" in html and "get_daily_series(metric=retention_d1)" in html
    assert "Rule-based summary" not in html


# --------------------------------------------------------------------------- validation
def test_claims_extraction_ignores_bare_integers():
    claims = [c[0] for c in claims_in("D1 retention fell 1.4pp over 12 weeks; CPM $15.20, -6.68 pts, 81%.")]
    assert claims == ["1.4pp", "$15.20", "-6.68 pts", "81%"]


def test_validation_is_unit_aware():
    book = NumberBook()
    book.add({"drivers": [{"contribution_pts": 1.24}, {"contribution_pts": 0.60}, {"contribution_pts": 0.54}],
              "cpm_change_pct": 2.4, "retention": "-1.7pp"})
    assert book.known(2.4, 1, "pct")              # a percentage we were given
    assert book.known(2.4, 1, "pts")              # 1.24 + 0.60 + 0.54: a legitimate sum of three drivers
    assert not book.known(2.5, 1, "pts")          # no points value or sum gives 2.5
    assert book.known(1.7, 1, "pp") and not book.known(1.7, 1, "pts")   # 1.7 appears only in pp
    assert not book.known(0.6, 1, "pct")          # 0.60 was points, not a percentage


def test_unknown_tool_does_not_use_budget(raw):
    r = result(raw, "US", "ios", max_tool_calls=1)
    out = run_agent(r, client=FakeClient([
        [tool_use(1, "x", {})],
        [tool_use(2, "get_weekly_trend", {"metric": "retention_d1"})],
        [tool_use(3, "submit_findings", GOOD_FINDINGS)]]))
    assert out.tool_calls == 1
    assert "Unknown tool 'x'" in out.trace[0].output["error"]
    assert "error" not in out.trace[1].output                 # the real call still had budget


def test_number_book_rounding_and_sums():
    book = NumberBook()
    book.add({"change_pct": -8.22, "drivers": [{"contribution_pts": -2.98}, {"contribution_pts": -1.26}]})
    assert book.known(8.2, 1) and book.known(8, 0)        # rounding
    assert book.known(4.24, 2) and book.known(4.2, 1)     # -2.98 + -1.26
    assert not book.known(8.5, 1)
    assert check_findings({"headline": "LTV fell 8.2%", "summary": "about 8.5%"}, book) != []


# --------------------------------------------------------------------------- tools on the planted world
def test_network_tool_shows_admob_dropout(raw):
    out = tools.execute(result(raw, "GB", "android"), "get_network_breakdown", {"ad_type": "Rewarded"})
    admob = next(n for n in out["networks"] if n["network"] == "AdMob")
    assert admob["share_this_pct"] == 0 and admob["mix_effect_pts"] < -5


def test_daily_series_reveals_tuesday_retention_step(raw):
    out = tools.execute(result(raw, "US", "android"), "get_daily_series", {"metric": "retention_d1"})
    current = {d["weekday"]: d["value"] for d in out["days"] if d["week"] == "current"}
    before = (current["Sun"] + current["Mon"]) / 2
    after = sum(current[d] for d in ("Tue", "Wed", "Thu", "Fri", "Sat")) / 5
    assert after < before * 0.93                         # planted: -13% from Tuesday


def test_market_comparison_shows_seasonal_cpm_rise(raw):
    out = tools.execute(result(raw, "DE", "ios"), "compare_cpm_across_markets", {"ad_type": "Interstitial"})
    assert len(out["markets"]) == 8
    assert 0 < out["median_change_other_markets_pct"] < 6   # Q4 ramp, everywhere


def test_tool_errors_are_returned_not_raised(raw):
    r = result(raw, "US", "ios")
    assert "error" in tools.execute(r, "get_daily_series", {"metric": "cpm"})          # missing ad_type
    assert "error" in tools.execute(r, "no_such_tool", {})


def test_briefing_is_valid_json_with_drivers(raw):
    b = build_briefing(result(raw, "US", "ios"))
    data = json.loads(b[b.find("{"):])
    assert data["drivers_points_of_ltv_change"][0]["driver"] == "Interstitial impressions per install, D0"


# --------------------------------------------------------------------------- malformed answers
def test_drivers_sent_as_json_string_are_repaired(raw):
    """The bug seen in a real run: drivers arrived as a JSON-encoded string, not an array."""
    r = result(raw, "US", "ios")
    as_string = {**GOOD_FINDINGS, "drivers": json.dumps(GOOD_FINDINGS["drivers"]),
                 "next_steps": "Check the release notes."}
    out = run_agent(r, client=FakeClient([[tool_use(1, "submit_findings", as_string)]]))
    assert out.findings["drivers"] == GOOD_FINDINGS["drivers"]
    assert out.findings["next_steps"] == ["Check the release notes."]
    assert out.warnings == []


def test_unusable_findings_are_sent_back_not_crashed(raw):
    r = result(raw, "US", "ios")
    broken = {**GOOD_FINDINGS, "drivers": "interstitials, mostly"}            # not JSON, not usable
    wrong_driver = {**GOOD_FINDINGS, "drivers": [{"driver": "Playtime", "explanation": "x",
                                                  "verdict": "specific_issue", "confidence": "high"}]}
    client = FakeClient([[tool_use(1, "submit_findings", broken)],
                         [tool_use(2, "submit_findings", wrong_driver)],
                         [tool_use(3, "submit_findings", GOOD_FINDINGS)]])
    out = run_agent(r, client=client)
    assert out.findings == GOOD_FINDINGS
    first_error = client.requests[1][-1]["content"][0]
    assert first_error["is_error"] and "must be an array of objects" in first_error["content"]
    second_error = client.requests[2][-1]["content"][0]
    assert "'Playtime' is not one of the drivers" in second_error["content"]
    html = render(build_context(r, out))
    assert "malformed, sent back" in html


# --------------------------------------------------------------------------- caching and cost
def test_prompt_caching_is_requested_and_costed(raw):
    r = result(raw, "US", "ios", prompt_caching=True)
    usage = {"input_tokens": 500, "output_tokens": 300, "cache_creation_input_tokens": 4000,
             "cache_read_input_tokens": 20000}
    client = FakeClient([[tool_use(1, "submit_findings", GOOD_FINDINGS)]], usage=usage)
    out = run_agent(r, client=client)
    assert client.kwargs[0]["extra_body"] == {"cache_control": {"type": "ephemeral"}}
    assert out.usage.total_input == 24500 and out.usage.cache_read == 20000
    # 500*2 + 300*10 + 4000*2.5 + 20000*0.2 = 1000 + 3000 + 10000 + 4000 = 18000 per M
    assert out.cost_usd == pytest.approx(0.018)
    assert out.uncached_cost_usd == pytest.approx((24500 * 2 + 300 * 10) / 1e6)
    assert out.cost_usd < out.uncached_cost_usd


def test_prompt_caching_can_be_switched_off(raw):
    r = result(raw, "US", "ios", prompt_caching=False)
    client = FakeClient([[tool_use(1, "submit_findings", GOOD_FINDINGS)]])
    run_agent(r, client=client)
    assert "extra_body" not in client.kwargs[0]
