"""Evaluation cases: one case = one market in one synthetic world, with its planted scenarios.

Two sets, kept apart on purpose:
- dev:  the hand-written world in synthetic.yaml. We look at these runs and tune on them.
- test: randomly generated worlds (random targets, networks, sizes, start days, controls), from
        a seed we don't look at while tuning. Scores on this set are the honest numbers.

Every planted scenario gets its TRUE effect size, measured with a counterfactual: the same world
is generated without any scenarios (identical noise, see synthetic.py), and the difference in the
expected driver's contribution is the scenario's effect in points of LTV change. Scenarios below
a materiality threshold are "minor": finding them is a bonus, missing them is not penalised -
a lead doesn't need to hear about a -0.3 pt effect.

True is not the same as visible. This week's noise can cancel part of a real effect, so each
scenario also gets its OBSERVED effect - what the data actually shows, at the level where the
scenario acts (the driver, or the network's price/mix effect). A material scenario whose observed
effect is too small or has the wrong sign is "undetectable": no analyst could find it from this
data, so missing it is reported separately instead of as an agent failure.
"""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.loader import from_raw
from src.data.synthetic import generate
from src.metrics.decomposition import decompose, driver_table
from src.metrics.weekly import build_weekly


@dataclass
class Scenario:
    id: str
    target: str
    ad_type: str | None
    network: str | None
    magnitude: float
    start: dt.date
    description: str
    expected_driver: str | None = None    # None: not a driver (e.g. install volume) -> not scored
    effect_pts: float | None = None       # true effect, from the counterfactual
    material: bool = False
    observed_pts: float | None = None     # what this week's data shows (true effect + noise)
    detectable: bool = False              # material AND visible in the data


@dataclass
class Case:
    set_name: str
    world: int
    geo: str
    platform: str
    week: pd.Timestamp
    raw: dict = field(repr=False)
    scenarios: list[Scenario] = field(default_factory=list)
    data_issues: list[dict] = field(default_factory=list)   # pipeline problems planted in this market

    @property
    def id(self) -> str:
        return f"{self.set_name}_w{self.world}_{self.geo}_{self.platform}"

    @property
    def is_control(self) -> bool:
        return not self.scenarios and not self.data_issues

    @property
    def scored(self) -> list[Scenario]:
        return [s for s in self.scenarios if s.expected_driver]


def expected_driver(target: str, ad_type: str | None) -> str | None:
    return {
        "imps_d0": f"{ad_type} impressions per install, D0",
        "imps_d1": f"{ad_type} impressions per D1 active, D1",
        "retention_d1": "D1 retention",
        "cpm": f"{ad_type} CPM",
        "network_share": f"{ad_type} CPM",   # a dropout shows up in the type's CPM (mix effect)
    }.get(target)


# --------------------------------------------------------------------------- random test worlds
def _sample_scenario(rng: np.random.Generator, cfg: dict, geo: str, platform: str,
                     week: dt.date, n: int) -> dict:
    target = rng.choice(["imps_d0", "imps_d1", "retention_d1", "cpm", "network_share"],
                        p=[0.25, 0.15, 0.20, 0.25, 0.15])
    ad_type = str(rng.choice(["Interstitial", "Rewarded"]))
    nets = cfg["networks"][ad_type]
    top3 = sorted(nets, key=lambda k: -nets[k]["share"])[:3]
    down = rng.random() < 0.7
    s = {"geo": geo, "platform": platform, "target": str(target),
         "start": week + dt.timedelta(days=int(rng.integers(0, 5)))}   # Sunday .. Thursday
    if target in ("imps_d0", "imps_d1"):
        s.update(ad_type=ad_type, magnitude=float(rng.uniform(0.75, 0.88) if down else rng.uniform(1.12, 1.22)))
    elif target == "retention_d1":
        s.update(magnitude=float(rng.uniform(0.85, 0.93) if down else rng.uniform(1.06, 1.10)))
    elif target == "cpm":
        s.update(ad_type=ad_type, network=str(rng.choice(top3)),
                 magnitude=float(rng.uniform(0.55, 0.75) if down else rng.uniform(1.35, 1.70)))
    else:
        s.update(ad_type=ad_type, network=str(rng.choice(top3)), magnitude=0.0)
    s["magnitude"] = round(s["magnitude"], 3)
    s["id"] = f"{geo}_{platform}_{target}_{n}"
    s["description"] = _describe(s)
    return s


def _describe(s: dict) -> str:
    day = s["start"].strftime("%a")
    what = {"imps_d0": f"{s.get('ad_type')} D0 impressions per install",
            "imps_d1": f"{s.get('ad_type')} D1 impressions per active",
            "retention_d1": "D1 retention",
            "cpm": f"{s.get('network')} {s.get('ad_type')} CPM",
            "network_share": f"{s.get('network')} stops serving {s.get('ad_type')}"}[s["target"]]
    if s["target"] == "network_share":
        return f"{what} from {day}"
    return f"{what} {(s['magnitude'] - 1) * 100:+.0f}% from {day}"


def random_world(base_cfg: dict, seed: int, control_share: float) -> dict:
    """A copy of the synthetic world with new noise (seed) and random scenarios per market."""
    rng = np.random.default_rng(seed)
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    week = cfg["last_week_start"]
    scenarios = []
    for combo in cfg["combos"]:
        if rng.random() < control_share:
            continue  # control: nothing planted
        k = 1 if rng.random() < 0.6 else 2
        used: set[str] = set()
        while len(used) < k:
            s = _sample_scenario(rng, cfg, combo["geo"], combo["platform"], week, len(scenarios))
            key = expected_driver(s["target"], s.get("ad_type"))
            if key in used:
                continue  # two scenarios on the same driver can't be scored separately
            used.add(key)
            scenarios.append(s)
    cfg["scenarios"] = scenarios
    return cfg


# --------------------------------------------------------------------------- building cases
def _observe(raw: dict, geo: str, platform: str, week: pd.Timestamp, n_weeks: int):
    """Driver contributions and factor-level contributions (network price/mix), in points."""
    data = from_raw(raw, geo, platform, week - pd.Timedelta(weeks=n_weeks + 1), week)
    weekly = build_weekly(data)
    dec = decompose(weekly, data.monetization, week)
    drivers = driver_table(dec, weekly).set_index("driver")["contribution_pts"]
    contrib = dec.contributions.assign(pts=dec.contributions["contribution_usd"] / dec.ltv_prev * 100)
    return drivers, contrib


def _observed_effect(s: "Scenario", drivers: pd.Series, contrib: pd.DataFrame) -> float:
    """What the agent can see: the network's own effect for network scenarios, else the driver."""
    if s.target in ("cpm", "network_share"):
        kind = "cpm_price" if s.target == "cpm" else "cpm_mix"
        rows = contrib[(contrib["kind"] == kind) & (contrib["ad_type"] == s.ad_type) & (contrib["network"] == s.network)]
        return float(rows["pts"].sum())
    return float(drivers.get(s.expected_driver, 0.0))


def build_cases(world_cfgs: list[dict], set_name: str, min_effect_pts: float, n_weeks: int,
                markets: set[tuple[str, str]] | None = None, detect_min_pts: float | None = None) -> list[Case]:
    detect_min_pts = min_effect_pts if detect_min_pts is None else detect_min_pts
    cases = []
    for w, cfg in enumerate(world_cfgs):
        week = pd.Timestamp(cfg["last_week_start"])
        raw = generate(cfg)
        counterfactual = generate({**cfg, "scenarios": []})
        for combo in cfg["combos"]:
            geo, platform = combo["geo"], combo["platform"]
            if markets and (geo, platform) not in markets:
                continue
            planted = [s for s in cfg.get("scenarios") or []
                       if s.get("geo") == geo and s.get("platform") == platform]
            issues = [d for d in cfg.get("data_issues") or [] if d["geo"] == geo and d["platform"] == platform]
            case = Case(set_name, w, geo, platform, week, raw, data_issues=issues)
            if planted:
                actual, actual_contrib = _observe(raw, geo, platform, week, n_weeks)
                base, _ = _observe(counterfactual, geo, platform, week, n_weeks)
            for s in planted:
                sc = Scenario(id=s["id"], target=s["target"], ad_type=s.get("ad_type"), network=s.get("network"),
                              magnitude=s["magnitude"], start=s["start"], description=s.get("description", ""),
                              expected_driver=expected_driver(s["target"], s.get("ad_type")))
                if sc.expected_driver:
                    sc.effect_pts = float(actual.get(sc.expected_driver, 0) - base.get(sc.expected_driver, 0))
                    sc.material = abs(sc.effect_pts) >= min_effect_pts
                    sc.observed_pts = _observed_effect(sc, actual, actual_contrib)
                    sc.detectable = (sc.material and abs(sc.observed_pts) >= detect_min_pts
                                     and (sc.observed_pts > 0) == (sc.effect_pts > 0))
                case.scenarios.append(sc)
            cases.append(case)
    return cases


def dev_cases(synthetic_cfg: dict, eval_cfg: dict, n_weeks: int, markets=None) -> list[Case]:
    return build_cases([synthetic_cfg], "dev", eval_cfg["min_effect_pts"], n_weeks, markets,
                       eval_cfg.get("detect_min_pts"))


def stress_cases(synthetic_cfg: dict, stress_cfg: dict, eval_cfg: dict, n_weeks: int, markets=None) -> list[Case]:
    """The hand-designed hard cases (stress.yaml) on top of the same game (synthetic.yaml)."""
    world = {**copy.deepcopy(synthetic_cfg), "seed": stress_cfg["seed"],
             "scenarios": stress_cfg.get("scenarios") or [], "data_issues": stress_cfg.get("data_issues") or []}
    return build_cases([world], "stress", eval_cfg["min_effect_pts"], n_weeks, markets,
                       eval_cfg.get("detect_min_pts"))


def holdout_cases(synthetic_cfg: dict, eval_cfg: dict, n_weeks: int, markets=None) -> list[Case]:
    worlds = [random_world(synthetic_cfg, eval_cfg["test_seed"] + i, eval_cfg["control_share"])
              for i in range(eval_cfg["test_worlds"])]
    return build_cases(worlds, "test", eval_cfg["min_effect_pts"], n_weeks, markets,
                       eval_cfg.get("detect_min_pts"))
