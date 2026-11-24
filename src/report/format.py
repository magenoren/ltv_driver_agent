"""Number formatting, shared by the console output and the HTML report."""
from __future__ import annotations

import math


def level(value: float, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return {"usd": f"${value:,.2f}", "usd4": f"${value:,.4f}", "pct": f"{value * 100:.1f}%",
            "num": f"{value:,.2f}", "int": f"{value:,.0f}"}[fmt]


def signed(value: float, unit: str = "", digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    value = round(value, digits) + 0.0  # avoid "−0.0"
    return f"{value:+,.{digits}f}{unit}".replace("-", "−")  # real minus sign reads better


def direction(value: float, dead_zone: float = 0.0) -> str:
    """CSS class for colouring a change. All our metrics are 'higher is better'."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or abs(value) <= dead_zone:
        return "flat"
    return "up" if value > 0 else "down"


def movement(change_pct: float) -> str:
    """'fell 8.2%' / 'rose 0.8%' / 'was flat (+0.4%)'"""
    if abs(change_pct) < 0.5:
        return f"was flat ({signed(change_pct, '%')})"
    return f"{'rose' if change_pct > 0 else 'fell'} {abs(change_pct):.1f}%"


def minus(text: str) -> str:
    """Typographic minus in already-formatted strings like '-18.1%'."""
    return text.replace("-", "−")
