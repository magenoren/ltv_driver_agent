"""Tests for the LMDI decomposition: the maths on toy numbers, then the planted scenarios."""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import from_raw
from src.data.synthetic import generate
from src.metrics.decomposition import decompose, driver_networks, driver_table, lmdi, log_mean
from src.metrics.monetization import network_breakdown
from src.metrics.weekly import build_weekly
from src.settings import load_yaml

WEEK = pd.Timestamp("2026-11-01")
PREV = WEEK - pd.Timedelta(days=7)


# --------------------------------------------------------------------------- LMDI maths
def test_log_mean():
    assert log_mean(2.0, 2.0) == 2.0
    assert log_mean(np.e, 1.0) == pytest.approx(np.e - 1)


def test_lmdi_single_factor_gets_everything():
    c = lmdi([{"x": (2.0, 3.0), "y": (5.0, 5.0)}])          # V: 10 -> 15, only x moved
    assert c["x"] == pytest.approx(5.0)
    assert c["y"] == pytest.approx(0.0)


def test_lmdi_offsetting_factors():
    c = lmdi([{"x": (2.0, 4.0), "y": (5.0, 2.5)}])          # x doubles, y halves: V unchanged
    assert c["x"] == pytest.approx(-c["y"])
    assert c["x"] > 0


def test_lmdi_is_exact_on_random_sums_of_products():
    rng = np.random.default_rng(0)
    terms = [{f"f{j}": tuple(rng.uniform(0.5, 2.0, 2)) for j in range(4)} for _ in range(10)]
    total = sum(np.prod([v[1] for v in t.values()]) - np.prod([v[0] for v in t.values()]) for t in terms)
    assert sum(lmdi(terms).values()) == pytest.approx(total)


def test_lmdi_handles_a_factor_going_to_zero():
    # A network's share goes to 0: the whole lost term is assigned to that share factor
    c = lmdi([{"share": (0.3, 0.0), "price": (10.0, 12.0)}])
    assert c["share"] == pytest.approx(-3.0, abs=1e-6)
    assert c["price"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- on the synthetic world
@pytest.fixture(scope="module")
def raw():
    return generate(load_yaml("synthetic.yaml"))


def run(raw, geo, platform):
    data = from_raw(raw, geo, platform, WEEK - pd.Timedelta(weeks=13), WEEK)
    weekly = build_weekly(data)
    return data, weekly, decompose(weekly, data.monetization, WEEK)


@pytest.mark.parametrize("geo,platform", [("US", "ios"), ("GB", "android"), ("BR", "ios")])
def test_contributions_add_up_to_the_mmp_change(raw, geo, platform):
    _, weekly, dec = run(raw, geo, platform)
    assert dec.contributions["contribution_usd"].sum() == pytest.approx(dec.change_usd, rel=1e-6)
    drivers = driver_table(dec, weekly)
    assert drivers["contribution_pts"].sum() == pytest.approx(dec.change_pct, rel=1e-6)


def test_model_is_close_to_mmp_level(raw):
    _, _, dec = run(raw, "US", "ios")
    assert abs(dec.model_this / dec.ltv_this - 1) < 0.10   # residual stays a minor factor


def test_us_ios_top_driver_is_d0_interstitials(raw):
    _, weekly, dec = run(raw, "US", "ios")
    top = driver_table(dec, weekly).iloc[0]
    assert top["driver"] == "Interstitial impressions per install, D0"
    assert top["share_of_change_pct"] > 60


def test_gb_android_offsetting_effects(raw):
    data, weekly, dec = run(raw, "GB", "android")
    drivers = driver_table(dec, weekly).set_index("driver")
    assert drivers.at["Interstitial CPM", "contribution_pts"] > 3
    rewarded_engagement = (drivers.at["Rewarded impressions per install, D0", "contribution_pts"]
                           + drivers.at["Rewarded impressions per D1 active, D1", "contribution_pts"])
    assert rewarded_engagement < -3
    assert drivers["share_of_change_pct"].isna().all()       # offsetting drivers: no "% of change"

    nb = network_breakdown(data.monetization, "Interstitial", WEEK, PREV)
    nets = driver_networks(dec, "Interstitial", 5, nb)
    assert nets["price_effect_pts"].idxmax() == "Mintegral"


def test_gb_android_admob_dropout_is_a_mix_effect(raw):
    data, _, dec = run(raw, "GB", "android")
    nb = network_breakdown(data.monetization, "Rewarded", WEEK, PREV)
    nets = driver_networks(dec, "Rewarded", 5, nb)
    assert nets.at["AdMob", "mix_effect_pts"] < 0           # lost AdMob's share...
    assert nets.at["AdMob", "price_effect_pts"] == pytest.approx(0, abs=1e-6)  # ...not a price change


def test_us_android_top_driver_is_retention(raw):
    _, weekly, dec = run(raw, "US", "android")
    assert driver_table(dec, weekly).iloc[0]["driver"] == "D1 retention"
