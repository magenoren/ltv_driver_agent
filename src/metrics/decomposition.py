"""Decompose the week-over-week change in LTVD1 (MMP) into its drivers, using LMDI.

The model
---------
Ad revenue per install over D0-1, summed over ad types t and networks n:

    LTV_model = sum_t sum_n [ x0_t  +  R1 * x1_t ] * s_tn * p_tn / 1000

    x0_t  = D0 impressions per install            (Firebase)
    R1    = D1 retention                          (Firebase)
    x1_t  = D1 impressions per D1 active user     (Firebase)
    s_tn  = network n's share of type-t impressions  (monetization table)
    p_tn  = network n's CPM for type t            (monetization table)

(sum_n s_tn * p_tn is exactly the type's CPM, so going down to networks loses nothing.)

The model uses all-user CPMs, so it never equals the MMP LTV exactly. We define
    k = LTV_MMP / LTV_model          (calibration factor, computed each week)
so that LTV_MMP = k * LTV_model holds exactly, and k becomes one more factor. Its contribution
is the "unexplained" part: cohort-vs-all-user pricing differences and measurement noise.

LMDI (Log Mean Divisia Index, additive form)
--------------------------------------------
For a term V = f1 * f2 * ... the change V1 - V0 is split as

    contribution(f) = L(V1, V0) * ln(f_this / f_prev),   L(a, b) = (a - b) / (ln a - ln b)

Because ln(V1/V0) = sum ln(f1/f0), the contributions add up EXACTLY to V1 - V0 - no
interaction residual, and the result doesn't depend on the order of factors (unlike
"change one factor at a time"). For a sum of terms, decompose each term and add up.

Zeros (a network that stopped serving): ln(0) is undefined. The usual fix is to replace 0 with a
tiny epsilon, but that converges very slowly (like 1/ln(epsilon)) and leaks a bit of the change
into the other factors. We use the exact limit instead: when a term goes to (or comes from) zero,
the whole change of that term belongs to the factor(s) that are zero.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.loader import AD_TYPES

EPS = 1e-12


# --------------------------------------------------------------------------- generic LMDI
def log_mean(a: float, b: float) -> float:
    if abs(a - b) < 1e-15:
        return a
    return (a - b) / (np.log(a) - np.log(b))


def lmdi(terms: list[dict[tuple, tuple[float, float]]]) -> dict[tuple, float]:
    """terms: each term is {factor_key: (prev_value, this_value)}; the term's value is the
    product of its factors. Returns {factor_key: contribution}, summing to the total change."""
    out: dict[tuple, float] = defaultdict(float)
    for factors in terms:
        v0 = np.prod([v[0] for v in factors.values()])
        v1 = np.prod([v[1] for v in factors.values()])
        zeros = [k for k, (a, b) in factors.items() if a <= EPS or b <= EPS]
        if zeros:  # term switches on/off: give its whole change to the zero factor(s)
            for k in factors:
                out[k] += (v1 - v0) / len(zeros) if k in zeros else 0.0
            continue
        weight = log_mean(v1, v0)
        for k, (a, b) in factors.items():
            out[k] += weight * np.log(b / a)
    return dict(out)


# --------------------------------------------------------------------------- inputs
def network_weekly(mon: pd.DataFrame) -> pd.DataFrame:
    """Per week x type x network: impression share and CPM."""
    g = mon.groupby(["date_week", "type", "network"])[["impressions", "revenue"]].sum()
    g["share"] = g["impressions"] / g.groupby(level=["date_week", "type"])["impressions"].transform("sum")
    g["cpm"] = g["revenue"] / g["impressions"] * 1000
    return g


def model_ltv(weekly: pd.DataFrame, week: pd.Timestamp) -> float:
    w = weekly.loc[week]
    return sum((w[f"{t.lower()}_d0_per_install"] + w["retention_d1"] * w[f"{t.lower()}_d1_per_active"])
               * w[f"cpm_{t.lower()}"] / 1000 for t in AD_TYPES)


# --------------------------------------------------------------------------- the decomposition
@dataclass
class Decomposition:
    week: pd.Timestamp
    ltv_prev: float
    ltv_this: float
    model_prev: float
    model_this: float
    contributions: pd.DataFrame   # one row per factor: kind, ad_type, network, contribution_usd

    @property
    def change_usd(self) -> float:
        return self.ltv_this - self.ltv_prev

    @property
    def change_pct(self) -> float:
        return (self.ltv_this / self.ltv_prev - 1) * 100


def decompose(weekly: pd.DataFrame, mon: pd.DataFrame, week: pd.Timestamp) -> Decomposition:
    prev = week - pd.Timedelta(days=7)
    nets = network_weekly(mon)
    w0, w1 = weekly.loc[prev], weekly.loc[week]
    m0, m1 = model_ltv(weekly, prev), model_ltv(weekly, week)
    k = (w0["ltv_d1_mmp"] / m0, w1["ltv_d1_mmp"] / m1)
    r1 = (w0["retention_d1"], w1["retention_d1"])

    terms = []
    for t in AD_TYPES:
        key = t.lower()
        x0 = (w0[f"{key}_d0_per_install"], w1[f"{key}_d0_per_install"])
        x1 = (w0[f"{key}_d1_per_active"], w1[f"{key}_d1_per_active"])
        n_prev = nets.loc[(prev, t)] if (prev, t) in nets.index.droplevel(2) else pd.DataFrame()
        n_this = nets.loc[(week, t)] if (week, t) in nets.index.droplevel(2) else pd.DataFrame()
        for n in sorted(set(n_prev.index) | set(n_this.index)):
            s = (n_prev["share"].get(n, 0.0), n_this["share"].get(n, 0.0))
            # A network absent in one week has no CPM that week; carry the other week's price
            # so its disappearance shows up as a share (mix) effect, not a fake price change.
            p_prev = n_prev["cpm"].get(n, np.nan)
            p_this = n_this["cpm"].get(n, np.nan)
            p = (p_prev if not np.isnan(p_prev) else p_this, p_this if not np.isnan(p_this) else p_prev)
            common = {("calibration", None, None): k,
                      ("cpm_mix", t, n): s,
                      ("cpm_price", t, n): p,
                      ("unit", None, None): (1 / 1000, 1 / 1000)}
            terms.append({**common, ("imps_d0", t, None): x0})                               # D0 revenue
            terms.append({**common, ("retention", None, None): r1, ("imps_d1", t, None): x1})  # D1 revenue

    contrib = lmdi(terms)
    rows = [{"kind": kind, "ad_type": t, "network": n, "contribution_usd": v}
            for (kind, t, n), v in contrib.items() if kind != "unit"]
    return Decomposition(week, w0["ltv_d1_mmp"], w1["ltv_d1_mmp"], m0, m1, pd.DataFrame(rows))


# --------------------------------------------------------------------------- tables for people
KIND_LABEL = {
    "retention": ("D1 retention", "Retention"),
    "imps_d0": ("{t} impressions per install, D0", "Engagement"),
    "imps_d1": ("{t} impressions per D1 active, D1", "Engagement"),
    "cpm": ("{t} CPM", "Pricing"),
    "calibration": ("Unexplained (cohort vs all-user pricing, measurement)", "Unexplained"),
}


def driver_table(dec: Decomposition, weekly: pd.DataFrame, min_change_for_share_pct: float = 1.0,
                 min_concentration: float = 0.5) -> pd.DataFrame:
    """One row per driver, biggest effect first. CPM price and mix are combined per ad type here;
    driver_networks() shows them per network.

    "% of the change" is only meaningful when the change is mostly in one direction. With
    offsetting drivers (+6 and -5 making -1), shares like -500% mislead, so we require both a
    minimum size and a minimum concentration: |total| / sum(|contributions|)."""
    c = dec.contributions.copy()
    c["driver_kind"] = c["kind"].replace({"cpm_mix": "cpm", "cpm_price": "cpm"})
    g = c.groupby(["driver_kind", "ad_type"], dropna=False)["contribution_usd"].sum().reset_index()

    prev = dec.week - pd.Timedelta(days=7)
    w0, w1 = weekly.loc[prev], weekly.loc[dec.week]

    def factor_change(kind: str, t) -> str:
        if kind == "retention":
            return f"{(w1['retention_d1'] - w0['retention_d1']) * 100:+.1f}pp"
        if kind == "calibration":
            k0, k1 = w0["ltv_d1_mmp"] / dec.model_prev, w1["ltv_d1_mmp"] / dec.model_this
            return f"{(k1 / k0 - 1) * 100:+.1f}%"
        col = {"imps_d0": "{}_d0_per_install", "imps_d1": "{}_d1_per_active", "cpm": "cpm_{}"}[kind]
        col = col.format(t.lower())
        return f"{(w1[col] / w0[col] - 1) * 100:+.1f}%"

    rows = []
    for _, r in g.iterrows():
        t = r["ad_type"] if isinstance(r["ad_type"], str) else None
        label, category = KIND_LABEL[r["driver_kind"]]
        rows.append({
            "driver": label.format(t=t),
            "category": category,
            "factor_change": factor_change(r["driver_kind"], t),
            "contribution_usd": r["contribution_usd"],
            # points of LTV % change: these add up to the total LTV % change
            "contribution_pts": r["contribution_usd"] / dec.ltv_prev * 100,
        })
    out = pd.DataFrame(rows)
    concentration = abs(out["contribution_usd"].sum()) / out["contribution_usd"].abs().sum()
    if abs(dec.change_pct) >= min_change_for_share_pct and concentration >= min_concentration:
        out["share_of_change_pct"] = out["contribution_usd"] / dec.change_usd * 100
    else:
        out["share_of_change_pct"] = np.nan
    return out.reindex(out["contribution_usd"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def driver_networks(dec: Decomposition, ad_type: str, min_share_pct: float, network_table: pd.DataFrame
                    ) -> pd.DataFrame:
    """Network-level price and mix effects for one ad type, in points of LTV change.
    network_table is the output of monetization.network_breakdown (defines the 'Other' grouping)."""
    c = dec.contributions[dec.contributions["ad_type"].eq(ad_type)
                          & dec.contributions["kind"].isin(["cpm_price", "cpm_mix"])].copy()
    shown = set(network_table.index) - {"Other"}
    c["network"] = c["network"].where(c["network"].isin(shown), "Other")
    pts = (c.pivot_table(index="network", columns="kind", values="contribution_usd", aggfunc="sum")
             .reindex(columns=["cpm_price", "cpm_mix"], fill_value=0) / dec.ltv_prev * 100)
    pts.columns = ["price_effect_pts", "mix_effect_pts"]
    return network_table.join(pts.reindex(network_table.index).fillna(0))
