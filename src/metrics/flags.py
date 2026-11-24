"""Which ad types deserve a closer look at the network level?

The type-level CPM can stay flat while networks move a lot underneath (AdMob drops out, the
others absorb its traffic at similar prices). So a type is flagged if ANY rule fires, and each
flag records the network and the reason, for the report and the agent.

Network rules look for network-SPECIFIC events. A network's price effect alone is not used:
in Q4 every network's CPM rises, and the biggest network then shows the biggest price effect
without anything having happened to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Flag:
    ad_type: str
    network: str | None     # None = a type-level rule
    reason: str


def network_flags(ad_type: str, type_cpm_change_pct: float, table: pd.DataFrame, th: dict) -> list[Flag]:
    """table: driver_networks() output for one ad type (shares, CPMs, price/mix effects)."""
    flags: list[Flag] = []
    if abs(type_cpm_change_pct) > th["cpm_drilldown_pct"]:
        flags.append(Flag(ad_type, None, f"{ad_type} CPM {type_cpm_change_pct:+.1f}%"))

    for network, r in table.drop(index="Other", errors="ignore").iterrows():
        big_enough = max(r["share_prev_pct"], r["share_this_pct"]) >= th["network_min_share_pct"]
        # Compare with the type's own CPM change: seasonality lifts every network together and
        # is not a network event. Mintegral +48% while the type moved +12% is.
        relative = r["cpm_change_pct"] - type_cpm_change_pct
        if big_enough and not pd.isna(relative) and abs(relative) > th["network_cpm_change_pct"]:
            flags.append(Flag(ad_type, network, f"{network} CPM {r['cpm_change_pct']:+.1f}% "
                                                f"(type {type_cpm_change_pct:+.1f}%)"))
        if abs(r["share_change_pp"]) > th["network_share_change_pp"]:
            flags.append(Flag(ad_type, network, f"{network} share {r['share_prev_pct']:.1f}% → "
                                                f"{r['share_this_pct']:.1f}%"))
        if abs(r["mix_effect_pts"]) > th["network_effect_pts"]:
            flags.append(Flag(ad_type, network, f"{network} mix effect {r['mix_effect_pts']:+.2f} pts"))
    return flags
