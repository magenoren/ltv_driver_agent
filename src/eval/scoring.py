"""Score one agent run against the planted ground truth, using only the structured findings.

A planted scenario counts as FOUND when the agent lists its driver with verdict "specific_issue".
Naming the driver but calling it noise or market-wide is a miss: the conclusion would be wrong.
For a network scenario we also record whether the agent named the right network.
A network dropout counts as found if any listed driver of that ad type names the network
(the agent may attribute it to CPM mix or to lost impressions; both are reasonable).

A FALSE ALARM is a driver listed as "specific_issue" that matches no planted scenario.
Listing drivers as normal variation or market-wide is fine - that's the agent showing its work.

In a market with a planted DATA ISSUE, the agent should say the data is the problem. This is
checked with keywords in its text (caveats, summary, explanations): a heuristic, because the
findings schema has no field for it. Calling the resulting metric moves a specific_issue is
already counted as a false alarm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.cases import Case, Scenario


@dataclass
class RunScore:
    found: dict[str, bool] = field(default_factory=dict)            # scenario id -> found
    network_named: dict[str, bool] = field(default_factory=dict)    # network scenarios only
    verdict_given: dict[str, str | None] = field(default_factory=dict)  # verdict on the scenario's driver
    false_alarms: list[str] = field(default_factory=list)
    false_alarm_confidence: list[str] = field(default_factory=list)   # "high" / "medium" / "low", same order
    artifact_recognized: bool | None = None   # None: no data issue planted in this market


def _text(entry: dict) -> str:
    return f"{entry.get('network') or ''} {entry.get('explanation') or ''}".lower()


def matches(entry: dict, s: Scenario) -> bool:
    driver = entry.get("driver", "")
    if s.target == "network_share":
        return bool(s.ad_type and s.ad_type in driver and s.network and s.network.lower() in _text(entry))
    return driver == s.expected_driver


ARTIFACT_WORDS = ("incomplete", "missing", "not matured", "immature", "not yet mature", "data quality",
                  "data issue", "data problem", "artifact", "partial data", "data gap", "not arrived",
                  "late data", "delayed data", "pipeline")


def recognizes_data_issue(findings: dict | None) -> bool:
    f = findings or {}
    texts = [f.get("headline", ""), f.get("summary", "")] + list(f.get("caveats") or [])
    texts += [d.get("explanation", "") for d in f.get("drivers", []) if isinstance(d, dict)]
    blob = " ".join(t for t in texts if isinstance(t, str)).lower()
    return any(w in blob for w in ARTIFACT_WORDS)


def score_run(case: Case, findings: dict | None) -> RunScore:
    score = RunScore()
    entries = [e for e in (findings or {}).get("drivers", []) if isinstance(e, dict)]
    specific = [e for e in entries if e.get("verdict") == "specific_issue"]

    for s in case.scored:
        hits = [e for e in specific if matches(e, s)]
        score.found[s.id] = bool(hits)
        any_mention = next((e for e in entries if matches(e, s)), None)
        score.verdict_given[s.id] = any_mention.get("verdict") if any_mention else None
        if s.network:
            score.network_named[s.id] = any(s.network.lower() in _text(e) for e in hits)

    for e in specific:
        if not any(matches(e, s) or e.get("driver") == s.expected_driver for s in case.scored):
            score.false_alarms.append(e.get("driver", "?"))
            score.false_alarm_confidence.append(e.get("confidence", "?"))
    if case.data_issues:
        score.artifact_recognized = recognizes_data_issue(findings)
    return score
