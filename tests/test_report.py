"""Tests for the network flags and the HTML report (on the synthetic world)."""
import pandas as pd
import pytest

from src.data.loader import from_raw
from src.data.synthetic import generate
from src.pipeline import analyze
from src.report.html_report import build_context, render, rule_based_summary, write_report
from src.settings import load_yaml

WEEK = pd.Timestamp("2026-11-01")


@pytest.fixture(scope="module")
def raw():
    return generate(load_yaml("synthetic.yaml"))


def result(raw, geo, platform):
    cfg = load_yaml("config.yaml")
    data = from_raw(raw, geo, platform, WEEK - pd.Timedelta(weeks=cfg["baseline_weeks"] + 1), WEEK)
    return analyze(data, geo, platform, WEEK, "csv", cfg)


def test_admob_dropout_is_flagged_although_rewarded_cpm_is_flat(raw):
    r = result(raw, "GB", "android")
    assert abs(r.type_cpm_change["Rewarded"]) < 10                  # the old rule would miss it
    assert any(f.network == "AdMob" and "share" in f.reason for f in r.flags_for("Rewarded"))
    assert any(f.network == "Mintegral" for f in r.flags_for("Interstitial"))


@pytest.mark.parametrize("geo,platform", [("DE", "ios"), ("DE", "android"), ("GB", "ios"), ("BR", "ios")])
def test_controls_raise_no_network_flags(raw, geo, platform):
    assert result(raw, geo, platform).flags == []


def test_summary_mentions_the_main_driver(raw):
    text = " ".join(rule_based_summary(result(raw, "US", "ios")))
    assert "fell 8.2%" in text and "Interstitial impressions per install, D0" in text


def test_report_renders_and_is_self_contained(raw, tmp_path):
    r = result(raw, "GB", "android")
    ctx = build_context(r)
    html = render(ctx)
    assert ctx["headline"] in html
    assert "<svg" in html and "Mintegral" in html
    assert "<script" not in html                                    # no JavaScript needed
    path = write_report(r, out_dir=tmp_path)
    assert path.exists() and path.stat().st_size > 10_000


def test_report_without_agent_shows_rule_based_summary(raw):
    ctx = build_context(result(raw, "US", "ios"), agent=None, agent_error="no API key")
    assert ctx["agent"] is None and ctx["agent_error"] == "no API key"
    assert "Rule-based summary" in render(ctx)
