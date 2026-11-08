"""Charts of the README, drawn from the tracked results in reports/review (run 20261104T1222Z).

* readme_contour.svg    - first draft vs the draft after verification vs the published answer:
                          correct answers, false refusals, hallucinations (significance.json);
* readme_verifiers.svg  - claim-level stress set, test split: correct claims kept against
                          distorted claims let through, per verifier (stress.json, baselines.json);
* readme_generators.svg - the verifier on the drafts of two generators (second_generator.json).

Palette: blue / orange / aqua slots of the dataviz reference palette (all-pairs CVD dE >= 9.2,
validated for scripts/article/figures.py) plus neutral grey; every value is labelled directly.
Text is converted to paths, the background is white, so GitHub light and dark themes render
the same picture.

Usage:  python scripts/readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reports" / "review"
OUT = ROOT / "reports" / "figures"
MM = 1 / 25.4
WIDTH = 180

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FALSE_KINDS = ("number_swap", "negation", "answer_swap", "transplant")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
        "mathtext.default": "regular",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((REVIEW / f"{name}.json").read_text("utf-8"))
    return data


def ru(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def sci(p: float) -> str:
    """5.6e-13 -> 5,6·10^-13 with a typeset exponent."""
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{ru(float(mantissa))}·$10^{{{int(exponent)}}}$"


def percent_axis(axis: Any) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))


def save(fig: Any, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="svg")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def contour() -> None:
    """Verification alone moves errors into refusals; the additional passes win them back."""
    part = load("significance")["parts"]["all"]
    stages = ["черновик", "после\nпроверки", "полный\nконтур"]
    colors = [GREY, GREY, BLUE]
    panels = [
        ("Верные ответы из 1307", [part["draft_vs_full"]["draft"],
                                   part["draft_vs_verified"]["verified"],
                                   part["draft_vs_full"]["full"]], 1307),
        ("Ложные отказы из 1038", [part["false_refusal"][k] for k in ("draft", "verified", "full")],
         1038),
        ("Галлюцинации из 269", [part["hallucination"][k]["k"]
                                 for k in ("draft", "verified", "full")], 269),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH * MM, 62 * MM),
                             gridspec_kw=dict(wspace=0.32))
    x = np.arange(3)
    for ax, (title, values, n) in zip(axes, panels, strict=True):
        ax.bar(x, values, width=0.62, color=colors, zorder=2)
        top = max(values) * 1.22 if max(values) > 5 else 6
        for xi, v in zip(x, values, strict=True):
            ax.text(xi, v + top * 0.02, f"{v}\n({ru(100 * v / n)} %)", ha="center", va="bottom",
                    fontsize=7)
        ax.set_ylim(0, top)
        ax.set_xticks(x, stages, fontsize=6.6)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, loc="left")
    pair = part["draft_vs_full"]
    fig.text(0.01, 0.015,
             f"Полный контур против черновика: +{pair['gained']}/−{pair['lost']} вопросов, "
             f"точный тест Мак-Немара p = {sci(pair['mcnemar_p'])}", fontsize=6.8, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.25)
    save(fig, "readme_contour.svg")


def verifiers() -> None:
    """Stress set, test split: share of correct claims kept against distorted claims accepted."""
    stress, rows = load("stress"), load("baselines")["rows"]
    series = [
        ("nli_single", "одиночный NLI-фильтр", GREY, True, "s", (-8, 0)),
        ("bge_zs", "BGE-M3-zeroshot", ORANGE, True, "o", (-8, 0)),
        ("bge_zs_dev", "BGE-M3-zeroshot", ORANGE, False, "o", (-8, -8)),
        ("minicheck", "MiniCheck", AQUA, True, "^", (-8, -3)),
        ("minicheck_dev", "MiniCheck", AQUA, False, "^", (-8, 8)),
        ("judge_free", "судья без якорей", BLUE, False, "D", (-8, 10)),
        ("judge", "FactRAG-NLI: судья + якоря", BLUE, True, "D", (-8, -10)),
    ]
    dev_tuned = {"bge_zs_dev", "minicheck_dev"}
    fig, ax = plt.subplots(figsize=(WIDTH * MM, 92 * MM))
    for label, name, color, filled, marker, (dx, dy) in series:
        rates = stress[label]
        false_k = sum(rates[f"{k}/test"]["k"] for k in FALSE_KINDS)
        false_n = sum(rates[f"{k}/test"]["n"] for k in FALSE_KINDS)
        x, y = 100 * rates["positive/test"]["rate"], 100 * false_k / false_n
        ax.scatter(x, y, s=52, marker=marker, color=color if filled else "white",
                   edgecolor=color, lw=1.3, zorder=3)
        thr = rows[label]["threshold"]
        if thr:
            name += f", порог {ru(thr, 2).rstrip('0')}" + (" с dev" if label in dev_tuned else "")
        text = f"{name}: {ru(x)} / {ru(y)}"
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha="right" if dx < 0 else "left", va="center")
    ax.scatter(100, 0, s=70, marker="*", color=INK2, zorder=3, clip_on=False)
    ax.annotate("идеал", (100, 0), xytext=(-6, 6), textcoords="offset points", ha="right",
                fontsize=7, color=INK2)
    ax.set_xlim(50, 102)
    ax.set_ylim(0, 70)
    percent_axis(ax.xaxis)
    percent_axis(ax.yaxis)
    n_true = stress["judge"]["positive/test"]["n"]
    n_false = sum(stress["judge"][f"{k}/test"]["n"] for k in FALSE_KINDS)
    ax.set_xlabel(f"Приняты верные утверждения, % (из {n_true})")
    ax.set_ylabel(f"Приняты искажённые утверждения, % (из {n_false})")
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_title("Стресс-набор, выборка test: чем левее и выше, тем хуже", loc="left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.13)
    save(fig, "readme_verifiers.svg")


def generators() -> None:
    """The same verifier on the drafts of Qwen2.5-14B-Instruct and Qwen3.8-27B (set v3)."""
    data = load("second_generator")
    gens = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    names = [k.split(" (")[0] for k, _ in gens]
    modes = [("vanilla", "черновик"), ("nli_single", "одиночный\nNLI-фильтр"),
             ("judge", "судья +\nякоря"), ("live", "живой прогон\n(с повторами)")]
    panels = [("Галлюцинации из 93", "hallucination"), ("Верные ответы из 220", "correct")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * MM, 66 * MM), gridspec_kw=dict(wspace=0.18))
    x = np.arange(len(modes))
    w = 0.38
    for ax, (title, key) in zip(axes, panels, strict=True):
        top = max(g[m][key] for _, g in gens for m, _ in modes) * 1.18
        for j, ((_, g), name, color) in enumerate(zip(gens, names, (ORANGE, BLUE), strict=True)):
            vals = [g[m][key] for m, _ in modes]
            pos = x + (j - 0.5) * w
            ax.bar(pos, vals, width=w - 0.04, color=color, label=name, zorder=2)
            for p, v in zip(pos, vals, strict=True):
                ax.text(p, v + top * 0.015, str(v), ha="center", va="bottom", fontsize=6.8)
        ax.set_ylim(0, top)
        ax.set_xticks(x, [m for _, m in modes], fontsize=6.8)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.2)
    save(fig, "readme_generators.svg")


if __name__ == "__main__":
    contour()
    verifiers()
    generators()
