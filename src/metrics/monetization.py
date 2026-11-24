"""CPM metrics from the monetization table (all users, not cohorted)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cpm_by_type(mon: pd.DataFrame) -> pd.DataFrame:
    """Weekly CPM per ad type: index date_week, columns Banner / Interstitial / Rewarded."""
    g = mon.groupby(["date_week", "type"])[["impressions", "revenue"]].sum()
    return (g["revenue"] / g["impressions"] * 1000).unstack("type")


def network_breakdown(mon: pd.DataFrame, ad_type: str, week: pd.Timestamp, prev: pd.Timestamp,
                      min_share_pct: float = 5.0) -> pd.DataFrame:
    """CPM and impression share per network for one ad type, previous vs current week.

    A network is shown on its own if it had at least min_share_pct of the type's impressions
    in EITHER week - so a network that stopped serving (share falls to 0) stays visible.
    The rest are grouped as "Other", so the table still adds up to 100%.
    """
    sub = mon[(mon["type"] == ad_type) & (mon["date_week"].isin([prev, week]))]
    g = (sub.groupby(["network", "date_week"])[["impressions", "revenue"]].sum()
            .unstack("date_week", fill_value=0))
    imps = g["impressions"].reindex(columns=[prev, week], fill_value=0)
    rev = g["revenue"].reindex(columns=[prev, week], fill_value=0)

    share = imps / imps.sum() * 100
    small = (share < min_share_pct).all(axis=1)
    label = pd.Series(np.where(small, "Other", imps.index), index=imps.index)
    imps = imps.groupby(label).sum()
    rev = rev.groupby(label).sum()
    share = imps / imps.sum() * 100

    cpm = (rev / imps.replace(0, np.nan)) * 1000  # no impressions -> CPM is undefined, not 0
    out = pd.DataFrame({
        "share_prev_pct": share[prev],
        "share_this_pct": share[week],
        "share_change_pp": share[week] - share[prev],
        "cpm_prev": cpm[prev],
        "cpm_this": cpm[week],
        "cpm_change_pct": (cpm[week] / cpm[prev] - 1) * 100,
        "impressions_this": imps[week],
    })
    out.index.name = "network"
    # Named networks by last week's share (a network that dropped out keeps its place), "Other" last
    order = out.drop(index="Other", errors="ignore").sort_values("share_prev_pct", ascending=False)
    if "Other" in out.index:
        order = pd.concat([order, out.loc[["Other"]]])
    return order
