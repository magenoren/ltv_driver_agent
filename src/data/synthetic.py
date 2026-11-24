"""Synthetic data generator for the three source tables.

The generator simulates one simple "world" and then measures it three ways, the way real
sources do:
  - MMP table          -> one row per install date: installs, cohort revenue D0 / D0-1
  - Firebase table     -> cohort events by age (0/1) and type, with ad revenue only on 'ad_revenue' rows
  - Monetization table -> daily impressions and revenue per network x ad type, for ALL users (not cohorted)

Because the three tables come from one world, they agree with each other except for deliberate,
realistic measurement differences (install counts, revenue gap, cohort vs all-user CPM).

Planted scenarios (synthetic.yaml) change the world on given dates. They are the "ground truth"
we later use to test whether the agent finds the real cause.

Counterfactual-friendly randomness: every market has its own random stream, and every random
draw happens whether or not a scenario applies. So generating the same world WITHOUT the
scenarios gives identical noise, and the difference between the two worlds is exactly the
planted effect (used by the evaluation to measure each scenario's true size).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

AD_TYPES = ["Banner", "Interstitial", "Rewarded"]


def week_start_sunday(d: dt.date) -> dt.date:
    """Sunday-based week start. Python's weekday(): Monday=0 ... Sunday=6."""
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def dow_index_sunday_first(d: dt.date) -> int:
    return (d.weekday() + 1) % 7


# --------------------------------------------------------------------------- scenarios
@dataclass
class ScenarioBook:
    """Holds the planted scenarios and answers: 'what multiplier applies here, on this date?'"""

    scenarios: list[dict]

    @staticmethod
    def _matches(scenario: dict, key: str, value) -> bool:
        # A scenario without e.g. 'network' applies to every network; with it, only to that one.
        wanted = scenario.get(key)
        return wanted is None or wanted == value

    def multiplier(self, target: str, geo: str, platform: str, date: dt.date,
                   ad_type: str | None = None, network: str | None = None) -> float:
        m = 1.0
        for s in self.scenarios:
            if s["target"] != target:
                continue
            if not (self._matches(s, "geo", geo) and self._matches(s, "platform", platform)
                    and self._matches(s, "ad_type", ad_type) and self._matches(s, "network", network)):
                continue
            start, end = s["start"], s.get("end")
            if date < start or (end is not None and date > end):
                continue
            ramp_days = s.get("ramp_days", 0)
            if ramp_days > 0:  # gradual change: move linearly from 1 to magnitude over ramp_days
                progress = min(1.0, ((date - start).days + 1) / ramp_days)
                m *= 1 + (s["magnitude"] - 1) * progress
            else:              # step change
                m *= s["magnitude"]
        return m


# --------------------------------------------------------------------------- generator
class SyntheticWorld:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])  # replaced per market in generate()
        self.scen = ScenarioBook(cfg.get("scenarios") or [])

        last = cfg["last_week_start"]
        first = last - dt.timedelta(weeks=cfg["n_weeks"] - 1)
        n_days = cfg["n_weeks"] * 7
        self.install_dates = [first + dt.timedelta(days=i) for i in range(n_days)]
        # D1 of the last cohort happens one day after the last install date
        self.activity_dates = self.install_dates + [self.install_dates[-1] + dt.timedelta(days=1)]

        # Normalise network price indices so the share-weighted average per type is exactly 1.
        self.net_norm = {
            t: sum(v["share"] * v["cpm_index"] for v in cfg["networks"][t].values()) for t in AD_TYPES
        }

    # ---- random helpers
    def noise(self, sigma: float) -> float:
        """Multiplicative log-normal noise with mean exactly 1 (the -sigma^2/2 term fixes the mean)."""
        return float(np.exp(self.rng.normal(0.0, sigma) - sigma ** 2 / 2))

    def sigma_for(self, installs_per_day: float) -> float:
        """Smaller samples are noisier: noise scales roughly with 1/sqrt(sample size)."""
        n = self.cfg["noise"]
        sigma = n["daily_sigma"] * np.sqrt(n["reference_installs"] / installs_per_day)
        return float(np.clip(sigma, n["daily_sigma"], n["max_sigma"]))

    def drift_series(self) -> np.ndarray:
        """Slow random walk (mean ~1) so baseline weeks wobble a little, like real data."""
        steps = self.rng.normal(0.0, self.cfg["noise"]["drift_sigma"], len(self.activity_dates))
        return np.exp(np.cumsum(steps))

    def seasonality(self, d: dt.date) -> float:
        s = self.cfg["seasonality"]
        span = (s["q4_peak"] - s["q4_start"]).days
        progress = np.clip((d - s["q4_start"]).days / span, 0.0, 1.0)
        return 1.0 + s["q4_uplift"] * float(progress)

    # ---- main
    def generate(self) -> dict[str, pd.DataFrame]:
        mmp_rows, fb_rows, mon_rows = [], [], []
        for i, combo in enumerate(self.cfg["combos"]):
            self.rng = np.random.default_rng([self.cfg["seed"], i])  # own stream per market
            cpm_by_type = self._monetization(combo, mon_rows)
            self._cohorts(combo, cpm_by_type, mmp_rows, fb_rows)

        mmp = pd.DataFrame(mmp_rows).sort_values(["country", "platform", "install_date"])
        fb = pd.DataFrame(fb_rows).sort_values(["country", "platform", "install_date", "age", "type"])
        mon = pd.DataFrame(mon_rows).sort_values(["country", "platform", "date", "type", "network"])
        tables = {"mmp": mmp.reset_index(drop=True),
                  "firebase": fb.reset_index(drop=True),
                  "monetization": mon.reset_index(drop=True)}
        return apply_data_issues(tables, self.cfg.get("data_issues") or [])

    def _monetization(self, combo: dict, rows: list) -> dict:
        """Daily all-user impressions/revenue per network x type. Returns the served CPM per (date, type)."""
        cfg, geo, platform = self.cfg, combo["geo"], combo["platform"]
        cpm_index = cfg["geo_cpm_index"][geo] * cfg["platform"][platform]["cpm_index"]
        dau_base = combo["installs_per_day"] * cfg["measurement"]["dau_per_daily_install"]
        sigma = self.sigma_for(dau_base / 5)  # all-user volumes are larger, so less noisy than cohorts
        cpm_by_type = {}

        for d in self.activity_dates:
            dau = dau_base * self.noise(0.02)
            dow_cpm = cfg["dow"]["cpm"][dow_index_sunday_first(d)]
            for t in AD_TYPES:
                nets = cfg["networks"][t]
                # 1) who serves the impressions: noisy shares, scenarios can switch a network off
                raw = {n: v["share"] * self.scen.multiplier("network_share", geo, platform, d, t, n)
                       * self.noise(0.05) for n, v in nets.items()}
                total_share = sum(raw.values())
                shares = {n: s / total_share for n, s in raw.items()}
                # 2) how many impressions of this type all users saw today
                type_imps = (dau * cfg["base"]["imps_d1_per_active"][t] * 0.9
                             * self.scen.multiplier("imps_d1", geo, platform, d, t) * self.noise(sigma))
                rev_sum, imp_sum = 0.0, 0
                for n, v in nets.items():
                    cpm_noise = self.noise(0.04)  # drawn even if unused, to keep streams aligned
                    if shares[n] <= 0:
                        continue  # network served nothing -> no row (as in real network reports)
                    cpm = (cfg["base"]["cpm_usd"][t] * cpm_index * v["cpm_index"] / self.net_norm[t]
                           * self.seasonality(d) * dow_cpm
                           * self.scen.multiplier("cpm", geo, platform, d, t, n) * cpm_noise)
                    imps = int(round(type_imps * shares[n]))
                    rev = imps * cpm / 1000
                    rows.append({"date": d, "date_week": week_start_sunday(d), "platform": platform,
                                 "country": geo, "network": n, "type": t,
                                 "impressions": imps, "revenue": round(rev, 4)})
                    rev_sum += rev
                    imp_sum += imps
                cpm_by_type[(d, t)] = rev_sum / imp_sum * 1000
        return cpm_by_type

    def _cohorts(self, combo: dict, cpm_by_type: dict, mmp_rows: list, fb_rows: list) -> None:
        cfg, geo, platform = self.cfg, combo["geo"], combo["platform"]
        base, meas = cfg["base"], cfg["measurement"]
        sigma = self.sigma_for(combo["installs_per_day"])
        drift = {"retention": self.drift_series()}
        for t in AD_TYPES:
            drift[f"imps_d0_{t}"] = self.drift_series()
            drift[f"imps_d1_{t}"] = self.drift_series()

        for i, d in enumerate(self.install_dates):
            week = week_start_sunday(d)
            # --- installs (MMP is the reference count; Firebase sees slightly fewer)
            n_mmp = int(round(combo["installs_per_day"] * cfg["dow"]["installs"][dow_index_sunday_first(d)]
                              * self.scen.multiplier("installs", geo, platform, d) * self.noise(sigma)))
            n_fb = int(round(n_mmp * meas["firebase_install_ratio"] * self.noise(0.01)))
            # --- D1 retention (by install date)
            r1 = (cfg["platform"][platform]["retention_d1"] * drift["retention"][i]
                  * self.scen.multiplier("retention_d1", geo, platform, d) * self.noise(sigma))
            a1 = int(round(n_fb * r1))

            fb_revenue = {0: 0.0, 1: 0.0}
            for age, active_users in ((0, n_fb), (1, a1)):
                act_date = d + dt.timedelta(days=age)
                common = {"install_date": d, "date_week": week, "platform": platform,
                          "country": geo, "age": age}
                other_events = int(round(active_users * base["other_events_per_user"][age] * self.noise(0.03)))
                fb_rows.append({**common, "type": "other", "users": active_users,
                                "events": other_events, "value": float(other_events)})

                target = "imps_d0" if age == 0 else "imps_d1"
                per_user_base = base["imps_d0_per_install"] if age == 0 else base["imps_d1_per_active"]
                total_imps, max_reach = 0, 0
                for t in AD_TYPES:
                    per_user = (per_user_base[t] * drift[f"{target}_{t}"][i + age]
                                * self.scen.multiplier(target, geo, platform, act_date, t) * self.noise(sigma))
                    events = int(round(active_users * per_user))
                    reach_noise = self.noise(0.02)  # drawn even if unused, to keep streams aligned
                    users = 0 if events == 0 else min(active_users, int(round(
                        active_users * base["reach"][t] * reach_noise)))
                    fb_rows.append({**common, "type": t, "users": users,
                                    "events": events, "value": float(events)})
                    # Cohort revenue: impressions x the CPM actually served that day, adjusted for new users
                    fb_revenue[age] += (events * cpm_by_type[(act_date, t)] / 1000
                                        * meas["cohort_cpm_premium"][age] * self.noise(0.02))
                    total_imps += events
                    max_reach = max(max_reach, users)
                fb_rows.append({**common, "type": "ad_revenue", "users": max_reach,
                                "events": total_imps, "value": round(fb_revenue[age], 4)})

            # --- MMP: same revenue, scaled to MMP installs, with a small stable gap
            scale = n_mmp / n_fb * meas["mmp_revenue_gap"]
            rev_d0 = fb_revenue[0] * scale * self.noise(0.005)
            rev_d1 = rev_d0 + fb_revenue[1] * scale * self.noise(0.005)
            mmp_rows.append({"install_date": d, "date_week": week, "platform": platform, "country": geo,
                             "installs": n_mmp, "cohort_rev_d0": round(rev_d0, 4),
                             "cohort_rev_d1": round(rev_d1, 4)})


def apply_data_issues(tables: dict[str, pd.DataFrame], issues: list[dict]) -> dict[str, pd.DataFrame]:
    """Pipeline problems, applied AFTER the world is generated: the world is fine, the data is not.

    missing_d1: the Firebase age=1 rows of one install date never arrived (the analysis ran before
    D1 data matured). Retention and D1 revenue look like they collapsed for that cohort.
    """
    for issue in issues:
        if issue["kind"] == "missing_d1":
            fb = tables["firebase"]
            drop = ((fb["country"] == issue["geo"]) & (fb["platform"] == issue["platform"])
                    & (fb["install_date"] == issue["install_date"]) & (fb["age"] == 1))
            tables["firebase"] = fb[~drop].reset_index(drop=True)
        else:
            raise ValueError(f"unknown data issue kind: {issue['kind']}")
    return tables


def generate(cfg: dict) -> dict[str, pd.DataFrame]:
    return SyntheticWorld(cfg).generate()
