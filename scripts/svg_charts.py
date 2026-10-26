"""Minimal stdlib SVG charts for the evaluation report (light and dark via prefers-color-scheme).

Palette: validated categorical slots (blue, orange, aqua, yellow) and a single-hue blue
ordinal ramp; light-mode aqua/yellow are below 3:1 contrast, so every bar carries a visible
value label and every chart has a table view in the report.
"""

from __future__ import annotations

from html import escape
from typing import Sequence

CATEGORICAL = ("--s1", "--s2", "--s3", "--s4")
ORDINAL = ("--o1", "--o2", "--o3", "--o4")
STYLE = """<style>
.viz{--surface:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--grid:#e4e3df;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--o1:#86b6ef;--o2:#5598e7;--o3:#2a78d6;--o4:#1c5cab;
--h0:#f0efec;--h1:#cde2fb;--h2:#86b6ef;--h3:#3987e5;--h4:#1c5cab;--h5:#0d366b;--inkhi:#ffffff}
@media (prefers-color-scheme: dark){.viz{--surface:#1a1a19;--text:#f0efec;--muted:#c3c2b7;
--grid:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--o1:#184f95;--o2:#2a78d6;--o3:#5598e7;--o4:#86b6ef;
--h0:#262624;--h1:#104281;--h2:#1c5cab;--h3:#2a78d6;--h4:#5598e7;--h5:#9ec5f4;--inkhi:#0b0b0b}}
text{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;fill:var(--text)}
.muted{fill:var(--muted)} .title{font-size:15px;font-weight:600} .lbl{font-size:12px}
.tick{font-size:11px} .grid{stroke:var(--grid);stroke-width:1}
</style>"""


def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="viz" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>{STYLE}"
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>{body}</svg>\n'
    )


def _bar(x: float, y: float, w: float, h: float, fill: str, vertical: bool) -> str:
    """Bar with a 4px rounded data end and a square baseline end."""
    if h <= 0 or w <= 0:
        return ""
    r = min(4.0, w / 2 if vertical else h / 2, h if vertical else w)
    if vertical:  # data end at the top
        d = (
            f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z"
        )
    else:  # data end at the right
        d = (
            f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z"
        )
    return f'<path d="{d}" fill="var({fill})"/>'


def _legend(items: Sequence[tuple[str, str]], x: float, y: float) -> str:
    out, cx = [], x
    for label, var in items:
        out.append(
            f'<rect x="{cx:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="var({var})"/>'
        )
        out.append(f'<text class="lbl" x="{cx + 14:.1f}" y="{y:.1f}">{escape(label)}</text>')
        cx += 24 + 7.2 * len(label)
    return "".join(out)


def grouped_bars(
    title: str,
    groups: Sequence[str],
    series: Sequence[str],
    values: Sequence[Sequence[float]],
    *,
    ordinal: bool = False,
    percent: bool = True,
    width: int = 760,
) -> str:
    """values[g][s]; vertical bars, shared 0..max axis, value label on every bar cap."""
    vars_ = ORDINAL if ordinal else CATEGORICAL
    height, top, bottom, left = 330, 64, 52, 46
    plot_h, plot_w = height - top - bottom, width - left - 16
    vmax = 1.0 if percent else max(max(row) for row in values) * 1.15
    band = plot_w / len(groups)
    bar_w = min(24.0, (band - 18) / len(series) - 2)
    body = [f'<text class="title" x="{left}" y="22">{escape(title)}</text>']
    body.append(_legend([(s, vars_[i]) for i, s in enumerate(series)], left, 44))
    for i in range(5):
        v = vmax * i / 4
        y = top + plot_h - plot_h * i / 4
        body.append(
            f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        tick = f"{v * 100:.0f}%" if percent else f"{v:.0f}"
        body.append(
            f'<text class="tick muted" x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>'
        )
    for g, group in enumerate(groups):
        cluster = len(series) * (bar_w + 2) - 2
        x0 = left + g * band + (band - cluster) / 2
        for s, value in enumerate(values[g]):
            h = plot_h * value / vmax
            x = x0 + s * (bar_w + 2)
            body.append(_bar(x, top + plot_h - h, bar_w, h, vars_[s], True))
            label = f"{value * 100:.0f}" if percent else f"{value:.0f}"
            body.append(
                f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{top + plot_h - h - 4:.1f}" '
                f'text-anchor="middle">{label}</text>'
            )
        body.append(
            f'<text class="lbl muted" x="{left + g * band + band / 2:.1f}" y="{height - 26}" '
            f'text-anchor="middle">{escape(group)}</text>'
        )
    return _svg(width, height, "".join(body), title)


def stacked_hbars(
    title: str,
    rows: Sequence[str],
    stages: Sequence[str],
    values: Sequence[Sequence[float]],
    unit: str = "мс",
    width: int = 760,
) -> str:
    """values[row][stage]; horizontal stacked bars with 2px surface gaps and totals."""
    left, top, row_h = 190, 64, 34
    height = top + row_h * len(rows) + 30
    plot_w = width - left - 90
    vmax = max(sum(r) for r in values) or 1.0
    body = [f'<text class="title" x="16" y="22">{escape(title)}</text>']
    body.append(_legend([(s, CATEGORICAL[i]) for i, s in enumerate(stages)], 16, 44))
    for r, (name, row) in enumerate(zip(rows, values, strict=True)):
        y = top + r * row_h
        body.append(
            f'<text class="lbl" x="{left - 8}" y="{y + 15}" text-anchor="end">{escape(name)}</text>'
        )
        x = float(left)
        for s, value in enumerate(row):
            w = plot_w * value / vmax
            if w <= 0:
                continue
            last = all(v <= 0 for v in row[s + 1 :])
            if last:
                body.append(_bar(x, y, max(w - 2, 0.5), 20, CATEGORICAL[s], False))
            else:
                body.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 0.5):.1f}" height="20" fill="var({CATEGORICAL[s]})"/>'
                )
            x += w
        body.append(
            f'<text class="tick" x="{x + 6:.1f}" y="{y + 14}">{sum(row):,.0f} {unit}</text>'.replace(
                ",", " "
            )
        )
    return _svg(width, height, "".join(body), title)


def heatmap(
    title: str,
    rows: Sequence[str],
    cols: Sequence[str],
    counts: Sequence[Sequence[int]],
    width: int = 420,
) -> str:
    """Row-normalised sequential heatmap with counts and row shares in every cell."""
    left, top, cell = 130, 74, 82
    height = top + cell * len(rows) + 16
    width = max(width, left + cell * len(cols) + 16)
    body = [f'<text class="title" x="16" y="22">{escape(title)}</text>']
    body.append(
        '<text class="tick muted" x="16" y="42">строка = эталон, столбец = прогноз; доля по строке</text>'
    )
    for c, col in enumerate(cols):
        body.append(
            f'<text class="tick muted" x="{left + c * cell + cell / 2:.1f}" y="{top - 8}" '
            f'text-anchor="middle">{escape(col)}</text>'
        )
    for r, row in enumerate(rows):
        total = sum(counts[r]) or 1
        y = top + r * cell
        body.append(
            f'<text class="lbl" x="{left - 8}" y="{y + cell / 2 + 4:.1f}" text-anchor="end">{escape(row)}</text>'
        )
        for c in range(len(cols)):
            share = counts[r][c] / total
            step = 0 if counts[r][c] == 0 else min(5, 1 + int(share * 5))
            body.append(
                f'<rect x="{left + c * cell + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2}" height="{cell - 2}" '
                f'rx="4" fill="var(--h{step})"/>'
            )
            ink = "var(--inkhi)" if step >= 3 else "var(--text)"
            body.append(
                f'<text class="lbl" x="{left + c * cell + cell / 2:.1f}" y="{y + cell / 2:.1f}" '
                f'text-anchor="middle" style="fill:{ink}">{counts[r][c]}</text>'
                f'<text class="tick" x="{left + c * cell + cell / 2:.1f}" y="{y + cell / 2 + 15:.1f}" '
                f'text-anchor="middle" style="fill:{ink}">{share * 100:.0f}%</text>'
            )
    return _svg(width, height, "".join(body), title)
