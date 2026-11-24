"""Small inline SVG charts, generated as strings.

Inline SVG keeps the report a single self-contained file: no chart library, no internet needed,
and it prints well. Colours come from CSS classes, so light/dark mode is handled by the page.
"""
from __future__ import annotations

import math
from html import escape


def sparkline(values: list[float], width: int = 96, height: int = 24) -> str:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad = 3
    step = (width - 2 * pad) / (len(vals) - 1)
    pts = [(pad + i * step, pad + (height - 2 * pad) * (1 - (v - lo) / span)) for i, v in enumerate(vals)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    last_x, last_y = pts[-1]
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="Trend over the last {len(vals)} weeks">'
            f'<polyline points="{path}" fill="none" stroke-width="1.5"/>'
            f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.5"/></svg>')


def bridge(rows: list[tuple[str, float]], total: float, unit: str = " pts") -> str:
    """Horizontal waterfall: each driver moves a running total from 0 to the total change.
    rows: (label, contribution) in display order."""
    row_h, top = 34, 8
    label_w, bar_x0, bar_w, value_w = 330, 340, 500, 100
    width = bar_x0 + bar_w + value_w
    n = len(rows) + 1
    height = top * 2 + n * row_h

    cum, spans = 0.0, []
    for _, v in rows:
        spans.append((cum, cum + v))
        cum += v
    points = [0.0, total] + [p for s in spans for p in s]
    lo, hi = min(points), max(points)
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad

    def x(v: float) -> float:
        return bar_x0 + (v - lo) / (hi - lo) * bar_w

    parts = [f'<svg class="bridge" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Contribution of each driver to the LTV change">']
    zero = x(0)
    parts.append(f'<line class="zero" x1="{zero:.1f}" y1="{top - 4}" x2="{zero:.1f}" y2="{height - top + 4}"/>')

    for i, ((label, v), (a, b)) in enumerate(zip(rows, spans)):
        y = top + i * row_h
        cls = "up" if v > 0 else "down"
        x1, x2 = sorted((x(a), x(b)))
        parts.append(f'<text class="label" x="0" y="{y + row_h / 2 + 4}">{escape(label)}</text>')
        parts.append(f'<rect class="{cls}" x="{x1:.1f}" y="{y + 7}" width="{max(x2 - x1, 1.5):.1f}" '
                     f'height="{row_h - 14}" rx="2"/>')
        value_txt = f"{v:+.2f}{unit}".replace("-", "−", 1)
        parts.append(f'<text class="value {cls}" x="{width}" y="{y + row_h / 2 + 4}" '
                     f'text-anchor="end">{value_txt}</text>')
        # connector to the next bar
        parts.append(f'<line class="connector" x1="{x(b):.1f}" y1="{y + row_h - 7}" '
                     f'x2="{x(b):.1f}" y2="{y + row_h + 7}"/>')

    y = top + len(rows) * row_h
    x1, x2 = sorted((x(0), x(total)))
    total_txt = f"{total:+.2f}{unit}".replace("-", "−", 1)
    parts.append(f'<text class="label total" x="0" y="{y + row_h / 2 + 4}">Total LTV change</text>')
    parts.append(f'<rect class="total" x="{x1:.1f}" y="{y + 7}" width="{max(x2 - x1, 1.5):.1f}" '
                 f'height="{row_h - 14}" rx="2"/>')
    parts.append(f'<text class="value total" x="{width}" y="{y + row_h / 2 + 4}" '
                 f'text-anchor="end">{total_txt}</text>')
    parts.append("</svg>")
    return "".join(parts)
