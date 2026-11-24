"""Data quality checks, run before any analysis. Each check returns human-readable issues;
an empty list means the data looks complete. The issues go into the report and to the agent,
so it knows when its conclusions rest on shaky data."""
from __future__ import annotations

import pandas as pd

from src.data.loader import AD_TYPES, SourceData


def check(data: SourceData, week: pd.Timestamp, prev: pd.Timestamp) -> list[str]:
    issues: list[str] = []
    for w in (prev, week):
        label = w.date().isoformat()
        for name, df, date_col in (("MMP", data.mmp, "install_date"),
                                   ("Firebase", data.firebase, "install_date"),
                                   ("Monetization", data.monetization, "date")):
            n_days = df.loc[df["date_week"] == w, date_col].nunique()
            if n_days != 7:
                issues.append(f"{name}: week {label} has {n_days}/7 days")

        fb_week = data.firebase[data.firebase["date_week"] == w]
        if (fb_week["installs_fb"] <= 0).any():
            issues.append(f"Firebase: week {label} has install dates with zero installs")
        if (fb_week["d1_actives"] <= 0).any():
            issues.append(f"Firebase: week {label} has install dates with no D1 users (D1 not matured?)")

        mon_week = data.monetization[data.monetization["date_week"] == w]
        for t in AD_TYPES:
            if mon_week.loc[mon_week["type"] == t, "impressions"].sum() <= 0:
                issues.append(f"Monetization: no {t} impressions in week {label}")
    return issues
