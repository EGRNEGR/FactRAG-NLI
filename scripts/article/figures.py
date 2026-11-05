"""Figures 1-8 of the article (Russian captions are in the manuscript, not in the images).

Data: reports/article/figure_data.json (scripts/article/figure_data.py). Output:
reports/article/figures/fig<N>.{png,pdf,svg}, 300 dpi, 170 mm wide (journal text width).

Style: Arial; decimal comma in every label; three validated categorical slots (blue, orange,
aqua: dataviz reference palette, all-pairs CVD dE >= 9.2) plus neutral grey; values are
labelled directly (aqua is below 3:1 contrast); print-safe secondary encoding - dashed
outlines for conditional passes, hatching on bars, distinct marker shapes.

Usage:  python scripts/article/figures.py [1 2 ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "reports" / "article" / "figure_data.json").read_text("utf-8"))
OUT = ROOT / "reports" / "article" / "figures"
MM = 1 / 25.4
WIDTH = 170

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_T, ORANGE_T, AQUA_T = "#e3eefb", "#fde8de", "#dff4ec"
GREY, GREY_T = "#6b6a66", "#f0efec"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def ru(x: float, digits: int = 1) -> str:
    """A number with a decimal comma."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def comma_axis(axis, digits: int = 1) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: ru(v, digits)))


def save(fig, n: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig{n}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("fig", n, "->", OUT / f"fig{n}.png")


# ------------------------------------------------------------------ diagram primitives


def canvas(height: float):
    fig = plt.figure(figsize=(WIDTH * MM, height * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, *, fc=GREY_T, ec=GREY, ls="-", fs=6.8, bold_first=False,
        lw=0.8, radius=1.2, align="center"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            fc=fc, ec=ec, lw=lw, ls=ls, zorder=2,
        )
    )
    lines = text.split("\n")
    if bold_first:
        gap = 1.3 * fs * 0.3528  # line pitch in mm
        top = y + h / 2 + gap * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            ax.text(x + w / 2, top - i * gap, line, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if i == 0 else "normal", zorder=3)
    else:
        tx = x + w / 2 if align == "center" else x + 1.5
        ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=fs, zorder=3,
                linespacing=1.25)


def diamond(ax, cx, cy, w, h, text, *, fc="white", ec=INK2, fs=6.6):
    ax.add_patch(
        Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
                closed=True, fc=fc, ec=ec, lw=0.8, zorder=2)
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.2)


def path(ax, points, *, color=INK2, ls="-", lw=0.9, head=True, label=None, label_at=None,
         label_offset=(0, 1.2), fs=6.3, label_ha="center"):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    if head:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                    color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=4)
    if label:
        i = label_at if label_at is not None else 0
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        ax.text((x0 + x1) / 2 + label_offset[0], (y0 + y1) / 2 + label_offset[1], label,
                ha=label_ha, va="center", fontsize=fs, color=color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.4))


def region(ax, x, y, w, h, title, *, ec, fc="none", ls=(0, (4, 2)), fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
                                fc=fc, ec=ec, lw=0.8, ls=ls, zorder=0))
    ax.text(x + 2, y + h - 2.2, title, ha="left", va="center", fontsize=fs,
            fontweight="bold", color=ec, zorder=1)


# ------------------------------------------------------------------ figure 1


def figure1() -> None:
    H = 160
    fig, ax = canvas(H)
    dash = (0, (3, 1.5))
    # Stage 1: retrieval (region x 33..163)
    region(ax, 33, 104, 130, 46, "Этап 1. Двухэтапный гибридный поиск (hybrid_retriever.py)",
           ec=BLUE, fc="#f6f9fe")
    box(ax, 2, 118, 27, 14, "Вопрос\nпользователя\n(RU или EN)", fc="white", ec=INK2,
        bold_first=True)
    box(ax, 37, 126, 28, 15, "BM25 (rank-bm25)\nдо 60 кандидатов;\nвес × доля терминов\nзапроса в словаре",
        fc=BLUE_T, ec=BLUE, bold_first=True, fs=6.2)
    box(ax, 37, 107, 28, 15, "BGE-M3 + Qdrant\nплотный поиск,\nдо 60 кандидатов",
        fc=BLUE_T, ec=BLUE, bold_first=True, fs=6.2)
    box(ax, 70, 117, 24, 15, "Reciprocal Rank\nFusion, k = 30;\n40 кандидатов",
        fc=BLUE_T, ec=BLUE, bold_first=True, fs=6.2)
    box(ax, 99, 117, 26, 15, "bge-reranker-v2-m3\nкросс-энкодер,\nокна 384 токена,\nсмесь α = 0,2",
        fc=BLUE_T, ec=BLUE, bold_first=True, fs=6.2)
    diamond(ax, 143, 124.5, 28, 17, "Гейт:\nmax s$_{CE}$ ≥ 0,005?", fs=6.2)
    path(ax, [(29, 125), (31, 125), (31, 133.5), (37, 133.5)])
    path(ax, [(31, 125), (31, 114.5), (37, 114.5)])
    path(ax, [(65, 133.5), (67.5, 133.5), (67.5, 126.5), (70, 126.5)])
    path(ax, [(65, 114.5), (67.5, 114.5), (67.5, 122.5), (70, 122.5)])
    path(ax, [(94, 124.5), (99, 124.5)])
    path(ax, [(125, 124.5), (129, 124.5)])
    box(ax, 131, 106, 24, 7.5, "топ-10 фрагментов", fc="white", ec=BLUE, fs=6.2)
    path(ax, [(143, 116), (143, 113.5)], color=BLUE)
    ax.text(144.5, 114.8, "да", fontsize=6.0, color=BLUE, va="center")

    # Stage 2: generation and verification
    region(ax, 33, 54, 130, 46, "", ec=ORANGE, fc="#fffaf7")
    ax.text(161, 97.8, "Этап 2. Генерация и независимая проверка", ha="right", va="center",
            fontsize=7.2, fontweight="bold", color=ORANGE)
    ax.text(161, 94.6, "generator_verifier.py, grounding.py", ha="right", va="center",
            fontsize=6.4, color=ORANGE)
    box(ax, 37, 69, 28, 22, "Qwen3.8-27B\nGGUF UD-Q4_K_XL,\nllama-server, T = 0;\nкаждое предложение\nсо ссылкой [S$_i$]",
        fc=ORANGE_T, ec=ORANGE, bold_first=True, fs=6.1)
    box(ax, 69, 69, 24, 22, "Декомпозиция\nна утверждения c$_j$,\nнаследование\nссылок",
        fc=ORANGE_T, ec=ORANGE, bold_first=True, fs=6.1)
    box(ax, 97, 74, 28, 17, "Судья (Judge)\nmDeBERTa-v3 +\nBGE-M3-zeroshot,\n13 признаков, θ = 0,3",
        fc=ORANGE_T, ec=ORANGE, bold_first=True, fs=6.1)
    box(ax, 97, 57, 28, 13, "Контроль якорей\nчисла, коды,\nидентификаторы",
        fc="white", ec=ORANGE, bold_first=True, fs=6.1)
    box(ax, 131, 74, 26, 17, "Политика допуска\nпубликация A,\nобрезка или отказ",
        fc=ORANGE_T, ec=ORANGE, bold_first=True, fs=6.1)
    path(ax, [(143, 106), (143, 102), (51, 102), (51, 91)], color=BLUE)
    path(ax, [(65, 80), (69, 80)])
    path(ax, [(93, 82.5), (97, 82.5)])
    path(ax, [(111, 70), (111, 74)])
    path(ax, [(125, 82.5), (131, 82.5)])
    box(ax, 136, 57, 25, 12, "Ответ с [S$_i$]\nили канонический\nотказ", fc="white", ec=INK,
        lw=1.1, fs=6.1)
    path(ax, [(148.5, 74), (148.5, 69)])

    # Adaptive passes (region y 15..50, boxes y 21..43)
    region(ax, 33, 15, 130, 35, "", ec=AQUA)
    ax.text(35, 17.8, "Адаптивные контуры (pipeline.py, query_translation.py)", ha="left",
            va="center", fontsize=7.2, fontweight="bold", color=AQUA)
    adaptive = [
        (37, "Рассуждение\nпосле отказа\nllm_think_retry"),
        (69, "Рассуждение\nпосле обрезки\nllm_think_on_trim:\nзамена только при\n100 % проверке"),
        (101, "Восстановление\n2 выборки, T = 0,7,\nесли max s$_{CE}$ ≥ 0,5"),
        (133, "G3 QUERY_\nTRANSLATION\nотказ гейта или LLM,\nразные алфавиты →\nперевод вопроса"),
    ]
    for x, text in adaptive:
        box(ax, x, 21, 28, 22, text, fc=AQUA_T, ec=AQUA, ls=dash, fs=6.0, bold_first=True)
    path(ax, [(131.5, 74), (131.5, 45)], color=AQUA, ls=dash)
    ax.text(130, 47.5, "отказ или обрезка", fontsize=6.0, color=AQUA, ha="right", va="center")
    path(ax, [(59, 43), (59, 69)], color=AQUA, ls=dash)
    ax.text(60.5, 52, "новый черновик", fontsize=6.0, color=AQUA, ha="left", va="center")
    # Gate refusal -> G3; G3 -> question
    path(ax, [(157, 124.5), (166, 124.5), (166, 36), (161, 36)], color=INK2, ls=dash)
    ax.text(159.2, 126.2, "нет", fontsize=6.0, color=INK2, ha="center")
    path(ax, [(161, 26), (169, 26), (169, 155), (15.5, 155), (15.5, 132)], color=AQUA, ls=dash)
    ax.text(92, 155, "переведённый вопрос: повторный поиск и один проверенный черновик",
            fontsize=6.2, color=AQUA, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4))

    # Diagnostics and levels
    box(ax, 2, 54, 27, 58,
        "Сквозная\nдиагностика\nRAGResponse:\nвремена стадий;\nоценки BM25,\nплотного поиска,\nRRF и кросс-\nэнкодера;\nчерновики\nи все попытки;\nP$_e$, P$_c$ обеих\nNLI-моделей;\nбалл судьи;\nпричина отказа",
        fc="white", ec=GREY, ls=(0, (1, 1.2)), bold_first=True, fs=6.0)
    levels = [
        "Конфигурация\nи ресурсы\nsettings.py",
        "Документный\nуровень\nparser v2",
        "Поиск, генерация\nи проверка",
        "Оркестрация\npipeline.py",
        "Интерфейсы\nи оценка\nevaluate_suite.py",
    ]
    ax.text(2, 7.5, "Функцио-\nнальные\nуровни", fontsize=6.3, fontweight="bold", va="center",
            color=INK2, linespacing=1.15)
    for i, text in enumerate(levels):
        box(ax, 20 + i * 29.8, 1.5, 27.8, 12, text, fc=GREY_T, ec=GREY, fs=6.0)
    save(fig, 1)


# ------------------------------------------------------------------ figure 2


def figure2() -> None:
    H = 92
    fig, ax = canvas(H)
    steps = [
        ("Файл\nTXT, MD, PDF\nс текстовым\nслоем, DOCX", "white", INK2),
        ("Проверка\nформат, размер,\n≤ 500 страниц;\nснимок файла", GREY_T, GREY),
        ("SHA-256\nидентификатор\nдокумента", GREY_T, GREY),
    ]
    x = 4
    for text, fc, ec in steps:
        box(ax, x, 62, 30, 22, text, fc=fc, ec=ec, bold_first=True, fs=6.6)
        x += 36
    path(ax, [(34, 73), (40, 73)])
    path(ax, [(70, 73), (76, 73)])
    diamond(ax, 136, 73, 34, 22, "Хеш уже\nв индексе?")
    path(ax, [(106, 73), (119, 73)])
    box(ax, 140, 40, 26, 13, "пропуск: векторы\nне пересчитываются", fc="white", ec=GREY,
        ls=(0, (3, 1.5)), fs=6.3)
    path(ax, [(153, 73), (160, 73), (160, 53)], label="да", label_at=0, label_offset=(0, 1.6))
    box(ax, 98, 28, 34, 30,
        "Структурный парсинг\n(parser v2)\nsection_path, страницы;\nтаблицы с подписью\nи шапкой; защита от\nложных заголовков",
        fc=BLUE_T, ec=BLUE, bold_first=True, fs=6.3)
    path(ax, [(136, 62), (136, 60.5), (115, 60.5), (115, 58)], label="нет", label_at=1, label_offset=(0, 1.6))
    box(ax, 52, 28, 38, 30,
        "Сегментация\nокна 320/384 токена\nBGE, перекрытие 50\nцелыми предложениями;\nшапка таблицы\nв каждом фрагменте",
        fc=BLUE_T, ec=BLUE, bold_first=True, fs=6.3)
    path(ax, [(98, 43), (90, 43)])
    box(ax, 4, 28, 40, 30,
        "Индексация\nэмбеддинги BGE-M3\n→ Qdrant; пересборка\nBM25; число\nфрагментов\nв метаданных",
        fc=BLUE_T, ec=BLUE, bold_first=True, fs=6.3)
    path(ax, [(52, 43), (44, 43)])
    box(ax, 4, 4, 70, 14,
        "Документ доступен поиску только после полной индексации;\nпрерванная индексация исключает его до повторной загрузки",
        fc="white", ec=INK, lw=1.0, fs=6.4)
    path(ax, [(24, 28), (24, 18)])
    box(ax, 84, 4, 82, 14,
        "Изменённое содержимое - новый хеш и новая версия документа;\nтот же файл под другим именем распознаётся по хешу",
        fc="white", ec=GREY, ls=(0, (1, 1.2)), fs=6.4)
    save(fig, 2)


# ------------------------------------------------------------------ figure 3


def figure3() -> None:
    H = 168
    fig, ax = canvas(H)
    dash = (0, (3, 1.5))
    ax.text(42, H - 3, "Уровень утверждения", fontsize=7.6, fontweight="bold", ha="center",
            color=ORANGE)
    ax.text(128, H - 3, "Уровень ответа и адаптивное восстановление", fontsize=7.6,
            fontweight="bold", ha="center", color=AQUA)
    ax.plot([86, 86], [4, H - 7], color=MUTED, lw=0.6, ls=(0, (2, 2)))

    # claim level (left column, centre x = 42)
    box(ax, 18, 146, 48, 11, "Черновик ответа (Vanilla):\nсегментация на утверждения c$_j$",
        fc="white", ec=INK2, fs=6.5)
    diamond(ax, 42, 133, 40, 13, "Есть ссылка [S$_i$]?")
    path(ax, [(42, 146), (42, 139.5)])
    box(ax, 68, 128, 14, 10, "UNVERIFIED:\nнет ссылки", fc="white", ec=GREY, fs=5.8)
    path(ax, [(62, 133), (68, 133)], label="нет", label_offset=(0, 1.7))
    diamond(ax, 42, 116, 44, 15, "Все якоря c$_j$ есть\nв Src(c$_j$) и заголовках?")
    path(ax, [(42, 126.5), (42, 123.5)], label="да", label_offset=(2.5, 0), label_ha="left")
    box(ax, 68, 111, 14, 10, "UNVERIFIED:\nanchor_not_\nin_source", fc="white", ec=GREY,
        fs=5.6)
    path(ax, [(64, 116), (68, 116)], label="нет", label_offset=(0, 1.7))
    box(ax, 12, 91, 60, 15,
        "Признаки φ по каждому источнику: P$_e$, P$_c$ mDeBERTa-v3\nи BGE-M3-zeroshot по фрагменту и окнам из 1-2\nпредложений; перекрытие, якоря, кодовые фрагменты",
        fc=ORANGE_T, ec=ORANGE, fs=6.0)
    path(ax, [(42, 108.5), (42, 106)], label="да", label_offset=(2.5, 0.2), label_ha="left")
    diamond(ax, 42, 77, 46, 15, "p = σ(β$_0$ + β$^T$φ) ≥ θ = 0,3\nдля процитированного S$_i$?")
    path(ax, [(42, 91), (42, 84.5)])
    box(ax, 68, 70, 14, 13, "UNVERIFIED:\nчастично или\nнедостаточно", fc="white", ec=GREY,
        fs=5.6)
    path(ax, [(65, 77), (68, 77)], label="нет", label_offset=(0, 1.7))
    diamond(ax, 42, 59, 46, 15, "Другой источник: P$_c$ ≥ 0,9\nи BGE-M3 не подтверждает?")
    path(ax, [(42, 69.5), (42, 66.5)], label="да", label_offset=(2.5, 0), label_ha="left")
    box(ax, 68, 54, 14, 10, "CONTRA-\nDICTION", fc="white", ec=GREY, fs=5.8)
    path(ax, [(65, 59), (68, 59)], label="да", label_offset=(0, 1.7))
    box(ax, 22, 38, 40, 11, r"VERIFIED: c$_j \in$ A", fc=ORANGE_T, ec=ORANGE, lw=1.0, fs=6.8)
    path(ax, [(42, 51.5), (42, 49)], label="нет", label_offset=(2.5, 0.2), label_ha="left")
    box(ax, 8, 14, 68, 16,
        "Совместная поддержка: UNVERIFIED по слабой поддержке\nпринимается, если источник подтвердил другое\nутверждение, все якоря на месте и перекрытие ≥ 0,5",
        fc="white", ec=ORANGE, ls=dash, fs=6.0)
    path(ax, [(42, 38), (42, 30)])
    path(ax, [(76, 22), (84.5, 22), (84.5, 156), (132, 156), (132, 151)])

    # answer level (right column)
    diamond(ax, 132, 141, 52, 20, "A = $\\varnothing$ или |C \\ A| / |C| > 0,5\nили генерация усечена?",
            fs=6.3)
    box(ax, 88, 104, 26, 16, "Отказ:\nllm_think_retry,\nзатем восстанов-\nление",
        fc=AQUA_T, ec=AQUA, ls=dash, fs=6.0)
    path(ax, [(106, 141), (101, 141), (101, 120)], label="да", label_at=0,
         label_offset=(0, 1.6))
    diamond(ax, 150, 118, 30, 14, "Снято ≥ 1\nутверждения?", fs=6.2)
    path(ax, [(132, 131), (132, 128), (150, 128), (150, 125)], label="нет", label_at=0,
         label_offset=(2.6, 0), label_ha="left")
    box(ax, 128, 92, 36, 14, "llm_think_on_trim:\nпроход think / think_repair\nс тем же контекстом",
        fc=AQUA_T, ec=AQUA, ls=dash, fs=6.0)
    path(ax, [(150, 111), (150, 106)], label="да", label_offset=(2.6, 0), label_ha="left")
    diamond(ax, 146, 77, 36, 16, "Все утверждения\nнового ответа\nVERIFIED?", fs=6.1)
    path(ax, [(146, 92), (146, 85)])
    box(ax, 120, 50, 22, 13, "прежний\nобрезанный\nответ", fc="white", ec=GREY, fs=6.0)
    path(ax, [(128, 77), (124, 77), (124, 63)], label="нет", label_at=0, label_offset=(0, 1.6))
    box(ax, 146, 50, 20, 13, "замена\nответа\nновым", fc=AQUA_T, ec=AQUA, fs=6.0)
    path(ax, [(164, 77), (160, 77)], head=False, color="none")
    path(ax, [(164, 77), (162, 77), (162, 63)], head=True, label="да", label_at=0,
         label_offset=(0.5, 1.6))
    box(ax, 88, 26, 28, 26,
        "G3 после отказа\nгейта или LLM:\nалфавиты вопроса\nи источников\nразличны →\nперевод, поиск,\nодин черновик",
        fc=AQUA_T, ec=AQUA, ls=dash, fs=6.0)
    path(ax, [(101, 104), (101, 52)], color=AQUA, ls=dash, label="снова\nотказ",
         label_offset=(0, 4), fs=6.0)
    box(ax, 120, 4, 46, 16,
        "Опубликованный ответ с цитатами [S$_i$]\nили канонический отказ",
        fc="white", ec=INK, lw=1.1, fs=6.3)
    path(ax, [(131, 50), (131, 20)])
    path(ax, [(156, 50), (156, 20)])
    path(ax, [(165, 118), (168.5, 118), (168.5, 12), (166, 12)])
    ax.text(167.4, 88, "нет: публикация A", rotation=90, fontsize=6.0, color=INK2,
            ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.3))
    path(ax, [(116, 39), (118, 39), (118, 12), (120, 12)])
    save(fig, 3)


# ------------------------------------------------------------------ figure 4


def strip_box(ax, data_list, positions, colors, *, width=0.55, jitter=0.17, seed=7,
              marker_size=5):
    rng = np.random.default_rng(seed)
    for values, pos, color in zip(data_list, positions, colors, strict=True):
        values = np.asarray(values)
        ax.boxplot(values, positions=[pos], widths=width, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=INK2, lw=0.7),
                   capprops=dict(color=INK2, lw=0.7),
                   boxprops=dict(facecolor="white", edgecolor=INK2, lw=0.7), zorder=2)
        x = pos + rng.uniform(-jitter, jitter, len(values))
        ax.scatter(x, values, s=marker_size, color=color, alpha=0.55, lw=0, zorder=3)


def figure4() -> None:
    gold = DATA["gold_scores"]
    e3 = DATA["translation_e3"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * MM, 78 * MM),
                               gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
    groups = [("en_other", "en_refused"), ("ru_other", "ru_refused")]
    positions, values, colors, labels = [], [], [], []
    for i, (other, refused) in enumerate(groups):
        for j, key in enumerate((other, refused)):
            positions.append(i * 2.4 + j)
            values.append(gold[key]["values"])
            colors.append(BLUE if j == 0 else ORANGE)
            labels.append(("остальные\n" if j == 0 else "упорные\nотказы\n") + f"n = {gold[key]['n']}")
    strip_box(a, values, positions, colors)
    for pos, key in zip(positions, ["en_other", "en_refused", "ru_other", "ru_refused"], strict=True):
        a.text(pos + 0.34, gold[key]["median"], ru(gold[key]["median"], 3), fontsize=7,
               va="center", ha="left", fontweight="bold",
               bbox=dict(fc="white", ec="none", pad=0.2))
    a.set_xticks(positions, labels)
    for x, lang in ((0.5, "Вопрос на английском"), (2.9, "Вопрос на русском")):
        a.text(x, -0.33, lang, ha="center", va="top", fontsize=7.5, fontweight="bold",
               transform=a.get_xaxis_transform())
    a.set_ylabel("Оценка реранкера для золотого фрагмента")
    a.set_ylim(-0.03, 1.05)
    comma_axis(a.yaxis, 1)
    a.set_xlim(-0.6, 4.2)
    a.set_title("а) XQuAD, прогон 1310Z: упорные отказы и остальные\nотвечаемые вопросы",
                loc="left")
    a.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)

    strip_box(b, [e3["before"], e3["after"]], [0, 1], [ORANGE, BLUE], marker_size=3)
    for pos, med in ((0, e3["median_before"]), (1, e3["median_after"])):
        b.text(pos + 0.34, med, ru(med, 3), fontsize=7, va="center", ha="left",
               fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.2))
    b.set_xticks([0, 1], ["исходный\nанглийский\nвопрос", "эталонный\nрусский\nперевод"])
    b.set_ylim(-0.03, 1.05)
    b.set_xlim(-0.6, 1.75)
    comma_axis(b.yaxis, 1)
    b.set_ylabel("Лучшая оценка реранкера")
    ga, g1 = e3["gate_answerable"], e3["gold_at_1"]
    n = e3["answerable"]
    b.set_title(f"б) E3: {n} отвечаемых EN-вопросов\nк русскому корпусу", loc="left")
    b.text(0.5, -0.33,
           f"гейт пройден: {ga['before']} → {ga['after']} из {n};\n"
           f"золотой фрагмент первым: {g1['before']} → {g1['after']}",
           ha="center", va="top", fontsize=7, transform=b.get_xaxis_transform(), color=INK2)
    b.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.27)
    save(fig, 4)


# ------------------------------------------------------------------ figure 5

TAXONOMY_RU = {
    "premise_not_in_context": "текст не утверждает спрошенного\n(десятилетие вместо года и т. п.)",
    "strict_premise_synonym": "строгое чтение синонима\nпредиката",
    "question_mt_garbled": "вопрос искажён машинным\nпереводом XQuAD",
    "needs_inference": "нужен вывод через несколько\nпредложений",
    "mt_lexical_gap": "межъязыковой лексический\nразрыв EN-RU",
    "dataset_noise": "ошибка эталона или перевода\nтекста",
    "question_not_standalone": "вопрос неполон вне\nконтекста статьи",
    "entity_alias": "разные имена одной\nсущности",
    "retrieval_miss": "промах поиска",
}
JUSTIFIED = {"premise_not_in_context", "dataset_noise", "question_not_standalone"}


def figure5() -> None:
    runs = DATA["runs"]
    order = [("0749Z", "0749Z\nисходный"), ("1310Z", "1310Z\n+ якоря,\n+ trim"),
             ("1929Z", "1929Z\n+ G3"), ("1222Z", "1222Z\n+ parser\nv2")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * MM, 92 * MM),
                               gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.75))
    x = np.arange(len(order))
    series = [("all", "все, 1307", BLUE, "o"), ("xquad-full", "XQuAD, 1087", AQUA, "s"),
              ("v3", "v3, 220", ORANGE, "^")]
    for key, label, color, marker in series:
        y = [100 * runs[r][key]["correct"][0] / runs[r][key]["correct"][1] for r, _ in order]
        a.plot(x, y, color=color, marker=marker, ms=4.5, lw=1.6, label=label, zorder=3,
               markeredgecolor="white", markeredgewidth=0.6)
    a.set_xticks(x, [label for _, label in order], fontsize=6.6)
    a.set_xlim(-0.35, len(order) - 0.65)
    a.set_ylim(86, 92)
    a.yaxis.set_major_locator(MultipleLocator(1))
    comma_axis(a.yaxis, 0)
    a.set_ylabel("Доля верных ответов FactRAG, %")
    a.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
    a.legend(loc="lower right", frameon=False, handlelength=2)
    a.set_title("а) Сквозная точность по рубежам", loc="left")
    # The values as a table under the axis: one row per series, one column per milestone.
    trans = a.get_xaxis_transform()
    for row, (key, label, color, marker) in enumerate(series):
        yy = -0.25 - 0.06 * row
        a.text(-0.42, yy, label.split(",")[0] + ", %", transform=trans, fontsize=6.6,
               color=color, ha="right", va="center", fontweight="bold")
        for i, (r, _) in enumerate(order):
            k, n = runs[r][key]["correct"]
            a.text(x[i], yy, ru(100 * k / n), transform=trans, fontsize=6.6, ha="center",
                   va="center", color=INK, fontweight="bold" if i == len(order) - 1 else None)

    tax = DATA["taxonomy"]["primary"]
    keys = list(tax)[::-1]
    y = np.arange(len(keys))
    vals = [tax[k] for k in keys]
    colors = [GREY_T if k in JUSTIFIED else BLUE_T for k in keys]
    edges = [GREY if k in JUSTIFIED else BLUE for k in keys]
    hatches = ["////" if k in JUSTIFIED else "" for k in keys]
    bars = b.barh(y, vals, color=colors, edgecolor=edges, lw=0.8, height=0.72, zorder=2)
    for bar, h in zip(bars, hatches, strict=True):
        bar.set_hatch(h)
    for yi, v in zip(y, vals, strict=True):
        b.text(v + 0.25, yi, str(v), va="center", fontsize=7.2)
    b.set_yticks(y, [TAXONOMY_RU[k] for k in keys], fontsize=6.6)
    b.set_xlim(0, 16)
    b.xaxis.set_major_locator(MultipleLocator(5))
    b.set_xlabel("Число вопросов")
    b.grid(axis="x", color="#e6e5e1", lw=0.5, zorder=0)
    justified = sum(tax[k] for k in JUSTIFIED)
    b.set_title(f"б) Причины {DATA['taxonomy']['cases']} упорных отказов XQuAD",
                loc="left")
    from matplotlib.patches import Patch

    b.legend(handles=[Patch(fc=GREY_T, ec=GREY, hatch="////",
                            label=f"обоснованный отказ или ошибка данных ({justified})"),
                      Patch(fc=BLUE_T, ec=BLUE,
                            label=f"потенциально устранимые ({DATA['taxonomy']['cases'] - justified})")],
             loc="upper center", bbox_to_anchor=(0.3, -0.16), frameon=False, fontsize=6.6)
    fig.subplots_adjust(left=0.1, right=0.975, top=0.92, bottom=0.33)
    save(fig, 5)


# ------------------------------------------------------------------ figure 6


def figure6() -> None:
    pts = DATA["final"]["points"]  # the final configuration, run 1222Z

    def kind(p):
        if p["status"] == "verified":
            return "verified"
        if p["reason"] == "anchor_not_in_source":
            return "anchor"
        if p["status"] == "contradiction":
            return "contradiction"
        return "low"

    styles = {
        "verified": ("допущены", BLUE, "o", 5, 0.35),
        "low": ("сняты: низкий балл судьи", ORANGE, "o", 9, 0.85),
        "contradiction": ("сняты: противоречие", GREY, "x", 12, 0.9),
        "anchor": ("сняты фильтром якорей", AQUA, "^", 16, 0.95),
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * MM, 84 * MM),
                               gridspec_kw=dict(wspace=0.3))
    for k in ("verified", "low", "contradiction", "anchor"):
        group = [p for p in pts if kind(p) == k]
        label, color, marker, size, alpha = styles[k]
        a.scatter([p["mdeberta_e"] for p in group], [p["bge_e"] for p in group], s=size,
                  color=color, marker=marker, alpha=alpha, lw=0.6 if marker == "x" else 0,
                  label=f"{label} ({len(group)})", zorder=3 if k != "verified" else 2)
    a.set_xlabel("P(entailment), mDeBERTa-v3")
    a.set_ylabel("P(entailment), BGE-M3-zeroshot")
    for axis in (a.xaxis, a.yaxis):
        comma_axis(axis, 1)
    a.set_xlim(-0.02, 1.02)
    a.set_ylim(-0.02, 1.02)
    a.set_title(f"а) {len(pts)} утверждений итогового прогона 1222Z", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(-0.12, -0.2), ncol=2, frameon=False,
             fontsize=6.6, handletextpad=0.3, columnspacing=1)
    a.grid(color="#efeeea", lw=0.5, zorder=0)

    # The residual cases of the discussion: a numbered ring, keyed below. Each is the withheld
    # claim of that question with the highest judge score (the one closest to publication).
    def case(cid, k=None):
        return max((p for p in pts if p["id"] == cid and p["marked"]
                    and (k is None or kind(p) == k)), key=lambda p: p["judge"])

    marks = [case("V3-MH-002"), case("V3-TC-021"), case("V3-TC-022", "anchor"),
             case("V3-TC-025", "anchor")]

    def ring(axis, x, y, n, dy=0.045):
        axis.scatter([x], [y], s=70, facecolors="none", edgecolors=INK, lw=0.9, zorder=6)
        # Near the right edge the number goes to the left (the title and counts are there).
        right = x > 0.9
        axis.text(x - 0.035 if right else x + 0.03, y - 3 * dy if right and dy < 0.1 else y + dy,
                  str(n), fontsize=7.2, fontweight="bold", zorder=7,
                  ha="right" if right else "left", va="bottom",
                  bbox=dict(fc="white", ec="none", pad=0.1))

    for n, p in enumerate(marks, start=1):
        ring(a, p["mdeberta_e"], p["bge_e"], n)

    rng = np.random.default_rng(3)
    order = ["verified", "low", "contradiction", "anchor"]
    for i, k in enumerate(order):
        vals = np.array([p["judge"] for p in pts if kind(p) == k])
        label, color, marker, size, alpha = styles[k]
        b.scatter(vals, i + rng.uniform(-0.28, 0.28, len(vals)), s=size, color=color,
                  marker=marker, alpha=alpha, lw=0.6 if marker == "x" else 0, zorder=3)
        b.text(1.03, i, f"n = {len(vals)}", va="center", fontsize=6.6, color=INK2)
    b.axvline(0.3, color=INK, lw=0.9, ls="-", zorder=2)
    b.axvline(0.15, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
    b.text(0.305, 3.62, "θ = 0,3", fontsize=6.6, ha="left")
    b.text(0.145, 3.62, "0,15", fontsize=6.6, ha="right", color=INK2)
    b.set_yticks(range(4), ["допущены", "низкий балл\nсудьи", "противоречие", "фильтр\nякорей"])
    b.set_ylim(-0.6, 3.85)
    b.set_xlim(-0.02, 1.0)
    comma_axis(b.xaxis, 1)
    b.set_xlabel("Балл судьи p = σ(β$_0$ + β$^T$φ)")
    b.set_title("б) Решение судьи и фильтр якорей", loc="left")
    b.grid(axis="x", color="#efeeea", lw=0.5, zorder=0)
    for n, p in enumerate(marks, start=1):
        row = order.index(kind(p))
        label, color, marker, size, alpha = styles[kind(p)]
        b.scatter([p["judge"]], [row], s=size * 1.6, color=color, marker=marker, zorder=5,
                  lw=0.6 if marker == "x" else 0)
        ring(b, p["judge"], row, n)
    key = (
        f"1 - V3-MH-002: разрешающее «MAY», P$_c$ mDeBERTa = {ru(marks[0]['mdeberta_c'], 2)}, "
        f"судья {ru(marks[0]['judge'], 3)}\n"
        f"2 - V3-TC-021: вся таблица как одна посылка, судья {ru(marks[1]['judge'], 3)}\n"
        f"3 - V3-TC-022: судья {ru(marks[2]['judge'], 3)} ≥ θ, но идентификатор\n"
        "      в соседнем фрагменте документа\n"
        f"4 - V3-TC-025: якорь «1000» против ячеек «1 | 0 | 0 | 0»"
    )
    fig.text(0.53, 0.02, key, fontsize=6.6, ha="left", va="bottom", linespacing=1.35)
    fig.subplots_adjust(left=0.075, right=0.92, top=0.9, bottom=0.36)
    save(fig, 6)


# ------------------------------------------------------------------ figure 7


def figure7() -> None:
    stage = DATA["final"]["stage_median_s"]  # the final configuration, run 1222Z
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * MM, 80 * MM),
                               gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.42))
    names = [("search_ms", "гибридный\nпоиск"), ("rerank_ms", "переранжи-\nрование"),
             ("nli_ms", "NLI-проверка\n(две модели)"), ("generation_ms", "генерация"),
             ("total_ms", "полный ответ")]
    y = np.arange(len(names))
    h = 0.36
    for j, (ds, label, color, hatch) in enumerate(
        [("v3", "v3", ORANGE, "////"), ("xquad-full", "XQuAD", BLUE, "")]
    ):
        vals = [stage[ds][k] for k, _ in names]
        bars = a.barh(y + (h / 2 if j == 0 else -h / 2), vals, height=h - 0.04,
                      color=ORANGE_T if j == 0 else BLUE_T, edgecolor=color, lw=0.8,
                      hatch=hatch, label=label, zorder=2)
        for v, yy in zip(vals, y + (h / 2 if j == 0 else -h / 2), strict=True):
            a.text(v + 0.08, yy, f"{ru(v, 2)} с", va="center", fontsize=6.6)
    a.set_yticks(y, [n for _, n in names], fontsize=7)
    a.set_xlim(0, 7.6)
    comma_axis(a.xaxis, 0)
    a.set_xlabel("Медиана, с (вопросы, дошедшие до генерации)")
    a.legend(loc="lower right", frameon=False)
    a.grid(axis="x", color="#e6e5e1", lw=0.5, zorder=0)
    a.set_title("а) Итоговый прогон 1222Z на RTX 3090", loc="left")

    # G3 is measured pairwise (1839Z against 1310Z, same questions); a single run cannot
    # separate the translated pass from the first one. The thinking pass is in the run's log.
    g3, trim = DATA["g3_cost"], DATA["final"]["think_on_trim"]
    strip_box(b, [g3["values_s"], trim["values_s"]], [0, 1], [AQUA, ORANGE], marker_size=7)
    b.set_xticks([0, 1], [f"G3: перевод и\nповторный поиск\n({g3['translated']} вопрос,\n1839Z к 1310Z)",
                          f"llm_think_on_trim\n({trim['calls']} вопросов,\n1222Z)"], fontsize=7)
    b.set_ylabel("Дополнительное время на вопрос, с")
    b.set_ylim(0, 46)
    comma_axis(b.yaxis, 0)
    b.set_xlim(-0.6, 1.6)
    b.text(0.34, g3["median_extra_s"] + 1.6,
           f"медиана {ru(g3['median_extra_s'])} с\nсреднее {ru(g3['mean_extra_s'])} с\n"
           f"всего +{ru(g3['total_extra_min'])} мин", fontsize=6.4, va="bottom",
           bbox=dict(fc="white", ec="none", pad=0.2))
    total_trim = trim["mean_s"] * trim["calls"] / 60
    b.text(0.62, 30,
           f"медиана {ru(trim['median_s'])} с\nсреднее {ru(trim['mean_s'])} с\n"
           f"всего +{ru(total_trim)} мин", fontsize=6.4, va="bottom",
           bbox=dict(fc="white", ec="none", pad=0.2))
    b.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
    b.set_title("б) Условные контуры: только на\nсработавших вопросах", loc="left")
    fig.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.2)
    save(fig, 7)


def figure8() -> None:
    """Verifiers on the claim-level stress set (test split): correct claims kept against
    distorted claims let through; then the distorted claims by kind."""
    rev = DATA["review"]
    # label, legend name, colour, fill, marker
    series = [
        ("nli_single", "одиночный NLI", GREY, True, "s"),
        ("bge_zs", "BGE-M3-zeroshot", ORANGE, True, "o"),
        ("bge_zs_dev", "BGE-M3-zeroshot (dev)", ORANGE, False, "o"),
        ("minicheck", "MiniCheck", AQUA, True, "^"),
        ("minicheck_dev", "MiniCheck (dev)", AQUA, False, "^"),
        ("judge_free", "судья без якорей", BLUE, False, "D"),
        ("judge", "судья + якоря (FactRAG-NLI)", BLUE, True, "D"),
    ]
    series = [s for s in series if s[0] in rev]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * MM, 82 * MM),
                               gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.34))
    offsets = {"nli_single": (6, -3), "bge_zs": (-6, 6), "bge_zs_dev": (-6, 6),
               "minicheck": (6, 4), "minicheck_dev": (6, -9), "judge_free": (-8, 7),
               "judge": (-8, -11)}
    for label, name, color, filled, marker in series:
        r = rev[label]
        x, y = 100 * r["true_accepted"], 100 * r["false_accepted"]
        a.scatter(x, y, s=46, marker=marker, color=color if filled else "white",
                  edgecolor=color, lw=1.2, zorder=3)
        dx, dy = offsets.get(label, (6, 4))
        thr = f", порог {ru(r['threshold'], 2)}" if r["threshold"] is not None else ""
        a.annotate(f"{name}{thr}", (x, y), xytext=(dx, dy), textcoords="offset points",
                   ha="right" if dx < 0 else "left", va="center", fontsize=6.3, color=INK)
    a.set_xlim(60, 102)
    a.set_ylim(0, 70)
    comma_axis(a.xaxis, 0)
    comma_axis(a.yaxis, 0)
    a.set_xlabel("Приняты верные утверждения, %")
    a.set_ylabel("Приняты искажённые утверждения, %")
    a.annotate("лучше", xy=(100, 3), xytext=(88, 15), fontsize=6.6, color=INK2,
               arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
    a.grid(color="#e6e5e1", lw=0.5, zorder=0)
    a.set_title("а) Компромисс: верные против искажённых", loc="left")

    kinds = [("number_swap", "подмена числа"), ("negation", "отрицание"),
             ("answer_swap", "подмена ответа\n(XQuAD)"), ("transplant", "утверждение из\nдругого документа")]
    shown = [s for s in series if s[0] in ("nli_single", "bge_zs", "minicheck", "judge")]
    hatches = {"nli_single": "xxxx", "bge_zs": "////", "minicheck": "....", "judge": ""}
    tints = {GREY: GREY_T, ORANGE: ORANGE_T, AQUA: AQUA_T, BLUE: BLUE_T}
    y = np.arange(len(kinds))[::-1]
    h = 0.8 / len(shown)
    for j, (label, name, color, _, _) in enumerate(shown):
        vals = [100 * rev[label]["by_kind"][k] for k, _ in kinds]
        pos = y + 0.4 - h / 2 - j * h
        b.barh(pos, vals, height=h - 0.03, color=tints[color], edgecolor=color, lw=0.7,
               hatch=hatches[label], label=name, zorder=2)
        for v, p in zip(vals, pos, strict=True):
            b.text(v + 1, p, ru(v), va="center", fontsize=5.9)
    b.set_yticks(y, [n for _, n in kinds], fontsize=7)
    b.set_xlim(0, 100)
    comma_axis(b.xaxis, 0)
    b.set_xlabel("Приняты искажённые утверждения, %")
    b.legend(loc="lower right", frameon=False, fontsize=6.4)
    b.grid(axis="x", color="#e6e5e1", lw=0.5, zorder=0)
    b.set_title("б) Искажения по видам", loc="left")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.15)
    save(fig, 8)


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6, 7: figure7,
           8: figure8}

if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or list(FIGURES):
        FIGURES[n]()
