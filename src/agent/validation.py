"""Guardrails for the agent's final answer.

1. Shape: normalize_findings() repairs a common quirk (arrays sent as JSON strings);
   structure_problems() lists what is still malformed, so it can be sent back to the model.
2. Numbers: every quantitative claim in the text must come from the briefing or a tool result.

Number checking is UNIT-AWARE. A claim like "+2.4 pts" may only match values the agent saw in
points (or a sum of up to three driver contributions); "18.1%" only percentages or unitless
values. Matching against every number regardless of unit makes accidental passes likely: the
more numbers the agent has seen, the more likely an invented one matches something.
Bare integers ("12 weeks", "D1", "Nov 4") are not checked.
"""
from __future__ import annotations

import json
import re
from itertools import combinations

NUMBER = re.compile(
    r"(?<![A-Za-z0-9.])"                      # not part of a word like D1 or a longer number
    r"(?P<dollar>\$)?[−\-+]?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>%|pp\b|pts?\b|points?\b)?"
)
UNITS = ("pts", "pp", "pct", "usd", "plain")
KEY_SUFFIX_UNIT = {"_pts": "pts", "_pp": "pp", "_pct": "pct", "_usd": "usd"}
# which kinds of known values a claim in each unit may match
COMPATIBLE = {"pts": ("pts",), "pp": ("pp", "plain"), "pct": ("pct", "plain"),
              "usd": ("usd", "plain"), "plain": UNITS}


def _unit_of_match(m: re.Match) -> str:
    if m.group("dollar"):
        return "usd"
    u = (m.group("unit") or "").lower()
    if u == "%":
        return "pct"
    if u == "pp":
        return "pp"
    if u.startswith("pt") or u.startswith("point"):
        return "pts"
    return "plain"


def _unit_of_key(key: str | None) -> str | None:
    if key:
        for suffix, unit in KEY_SUFFIX_UNIT.items():
            if key.endswith(suffix):
                return unit
    return None


def claims_in(text: str) -> list[tuple[str, float, int, str]]:
    """Quantitative claims in free text: (as written, value, decimals, unit)."""
    claims = []
    for m in NUMBER.finditer(text):
        num = m.group("num").replace(",", "")
        if not (m.group("dollar") or m.group("unit") or "." in num):
            continue  # bare integer: counts, days, ids - not checked
        decimals = len(num.split(".")[1]) if "." in num else 0
        claims.append((m.group(0).strip(), float(num), decimals, _unit_of_match(m)))
    return claims


class NumberBook:
    """The numbers the agent has legitimately seen, by unit."""

    def __init__(self):
        self.values: dict[str, set[float]] = {u: set() for u in UNITS}
        self.driver_pts: list[float] = []   # signed driver contributions (briefing)
        self.all_pts: list[float] = []      # every signed value in points

    def add(self, obj) -> None:
        self._walk(obj, None)

    def _put(self, value: float, unit: str) -> None:
        self.values[unit].add(abs(value))
        if unit == "pts":
            self.all_pts.append(value)

    def _walk(self, obj, key: str | None, unit_hint: str | None = None) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            unit = unit_hint or _unit_of_key(key) or "plain"
            self._put(float(obj), unit)
            if key == "contribution_pts":
                self.driver_pts.append(float(obj))
        elif isinstance(obj, str):
            for m in NUMBER.finditer(obj):
                value = float(m.group("num").replace(",", ""))
                sign = -1.0 if re.match(r"\$?[−-]", m.group(0)) else 1.0
                self._put(sign * value, _unit_of_match(m))
        elif isinstance(obj, dict):
            # {"change": -1.4, "change_unit": "pp"}: the unit of "change" is given by its sibling
            change_unit = {"%": "pct", "pp": "pp"}.get(obj.get("change_unit"))
            for k, v in obj.items():
                self._walk(v, k, change_unit if k == "change" else None)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._walk(v, key)

    def known(self, value: float, decimals: int, unit: str = "plain") -> bool:
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        for u in COMPATIBLE[unit]:
            if any(abs(v - value) <= tol for v in self.values[u]):
                return True
        if unit in ("pts", "plain"):
            # adding up contributions is allowed: up to three drivers, or any two point values
            for n in (2, 3):
                if any(abs(abs(sum(c)) - value) <= tol for c in combinations(self.driver_pts, n)):
                    return True
            if any(abs(abs(a + b) - value) <= tol for a, b in combinations(self.all_pts, 2)):
                return True
        return False


LIST_FIELDS = ("drivers", "next_steps", "caveats")
CONFIDENCE = ("high", "medium", "low")
VERDICTS = ("specific_issue", "market_wide", "normal_variation")


def normalize_findings(findings: dict) -> dict:
    """Repair a common model quirk: an array field sent as a JSON-encoded string.
    '[{"driver": ...}]' -> [{"driver": ...}];  'Check X' -> ['Check X'] for text lists."""
    out = dict(findings)
    for key in LIST_FIELDS:
        value = out.get(key)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                out[key] = parsed
            elif key != "drivers" and value.strip():
                out[key] = [value]
    if isinstance(out.get("drivers"), list):  # items can be encoded strings too
        items = []
        for item in out["drivers"]:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    pass
            items.append(item)
        out["drivers"] = items
    return out


def structure_problems(findings: dict, driver_names: list[str]) -> list[str]:
    """Shape errors the schema should have prevented but the model can still produce."""
    problems = []
    for key in ("headline", "summary"):
        if not isinstance(findings.get(key), str) or not findings[key].strip():
            problems.append(f"'{key}' must be a non-empty string")
    drivers = findings.get("drivers")
    if not isinstance(drivers, list):
        problems.append("'drivers' must be an array of objects (not a string)")
    else:
        for i, d in enumerate(drivers):
            if not isinstance(d, dict):
                problems.append(f"drivers[{i}] must be an object with driver, explanation, verdict, confidence")
                continue
            if d.get("driver") not in driver_names:
                problems.append(f"drivers[{i}].driver '{d.get('driver')}' is not one of the drivers in the briefing")
            if not isinstance(d.get("explanation"), str):
                problems.append(f"drivers[{i}].explanation must be a string")
            if d.get("verdict") not in VERDICTS:
                problems.append(f"drivers[{i}].verdict must be one of {list(VERDICTS)}")
            if d.get("confidence") not in CONFIDENCE:
                problems.append(f"drivers[{i}].confidence must be one of {list(CONFIDENCE)}")
    for key in ("next_steps", "caveats"):
        value = findings.get(key, [])
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            problems.append(f"'{key}' must be an array of strings")
    return problems


def check_findings(findings: dict, book: NumberBook) -> list[str]:
    """Return a list of unverified numbers, as messages the agent can act on."""
    texts = [findings.get("headline", ""), findings.get("summary", "")]
    texts += [d.get("explanation", "") for d in findings.get("drivers", []) if isinstance(d, dict)]
    texts += list(findings.get("next_steps", [])) + list(findings.get("caveats", []))
    problems = []
    for text in (t for t in texts if isinstance(t, str)):
        for written, value, decimals, unit in claims_in(text):
            if not book.known(value, decimals, unit):
                problems.append(f"'{written}' in: \"{text[:120]}{'…' if len(text) > 120 else ''}\"")
    return problems


def briefing_numbers(briefing_text: str) -> dict:
    """The briefing is text + JSON; parse the JSON part so its numbers enter the book."""
    start = briefing_text.find("{")
    return json.loads(briefing_text[start:]) if start >= 0 else {}
