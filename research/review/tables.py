"""Markdown tables of the article built from the review results (reports/review/*.json).

  Table 5  outcomes of the first drafts of 1222Z under each verifier (baselines.json)
  Table 6  acceptance of correct and distorted claims, stress set, test split (stress.json)
  Table 7  the verifier on two generators, v3 first drafts (second_generator.json)

Decimal comma, counts as in the rest of the article. Usage:  python -m research.review.tables
"""

from __future__ import annotations

import json
from typing import Any

from research.review import OUT

NAMES = {
    "vanilla": ("черновик без проверки (Vanilla)", "–"),
    "nli_single": ("одиночный NLI-фильтр (весь фрагмент)", "0,40 / 0,20"),
    "bge_zs": ("BGE-M3-zeroshot без якорей", None),
    "bge_zs_dev": ("BGE-M3-zeroshot без якорей", None),
    "minicheck": ("MiniCheck-Flan-T5-Large", None),
    "minicheck_dev": ("MiniCheck-Flan-T5-Large", None),
    "judge_free": ("судья без правила якорей", "0,3"),
    "judge": ("судья и контроль якорей (FactRAG-NLI)", "0,3"),
}
KINDS = (
    ("positive", "Верные"),
    ("number_swap", "Подмена числа"),
    ("negation", "Отрицание"),
    ("answer_swap", "Подмена ответа"),
    ("transplant", "Утверждение из другого документа"),
)


def ru(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def p_value(p: float) -> str:
    if p >= 0.001:
        return f"{p:.2g}".replace(".", ",")
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"${mantissa.replace('.', '{,}')} \\cdot 10^{{{int(exponent)}}}$"


def threshold(label: str, row: dict[str, Any]) -> str:
    fixed = NAMES[label][1]
    if fixed is not None:
        return fixed
    t = ru(row["threshold"], 2).rstrip("0").rstrip(",")
    return f"{t} (dev)" if label.endswith("_dev") else t


def table5(full: dict[str, Any]) -> str:
    rows = json.loads((OUT / "baselines.json").read_text("utf-8"))["rows"]
    lines = [
        "| Метод проверки | Порог | Верно, все | Верно, v3 | Верно, XQuAD | Галлюцинации (из 269) "
        "| Ложные отказы (из 1038) | Попарно к черновику |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, r in rows.items():
        a = r["all"]
        p = a["paired_vs_vanilla"]
        paired = "–" if label == "vanilla" else f"+{p['gained']}/−{p['lost']}"
        lines.append(
            f"| {NAMES[label][0]} | {threshold(label, r)} | {a['correct']} | {r['v3']['correct']} "
            f"| {r['xquad-full']['correct']} | {a['hallucination']} | {a['false_refusal']} | {paired} |"
        )
    lines.append(
        f"| полный контур FactRAG-NLI (опубликованный ответ) | 0,3 | {full['correct']} "
        f"| {full['v3']} | {full['xquad']} | {full['hallucination']} | {full['false_refusal']} "
        f"| +{full['gained']}/−{full['lost']} |"
    )
    return "\n".join(lines)


def table6() -> str:
    stress = json.loads((OUT / "stress.json").read_text("utf-8"))
    rows = json.loads((OUT / "baselines.json").read_text("utf-8"))["rows"]
    labels = [k for k in rows if k in stress]
    first = stress[labels[0]]
    head = " | ".join(f"{name} ({first[f'{kind}/test']['n']})" for kind, name in KINDS)
    false = [k for k, _ in KINDS[1:]]
    false_n = sum(first[f"{k}/test"]["n"] for k in false)
    lines = [
        f"| Метод проверки | Порог | {head} | Все искажения ({false_n}) |",
        "|---|---|" + "---|" * (len(KINDS) + 1),
    ]
    for label in labels:
        s = stress[label]
        cells = " | ".join(ru(100 * s[f"{kind}/test"]["rate"]) for kind, _ in KINDS)
        pooled = sum(s[f"{k}/test"]["k"] for k in false) / false_n
        lines.append(
            f"| {NAMES[label][0]} | {threshold(label, rows[label])} | {cells} | {ru(100 * pooled)} |"
        )
    return "\n".join(lines)


def table7() -> str:
    data = json.loads((OUT / "second_generator.json").read_text("utf-8"))
    modes = {
        "vanilla": "черновик без проверки",
        "nli_single": "одиночный NLI-фильтр",
        "judge": "судья и контроль якорей",
        "live": "живой прогон (с повторными проходами)",
    }
    lines = [
        "| Генератор | Режим | Верно (из 220) | Галлюцинации (из 93) | Ложные отказы (из 127) "
        "| Попарно к черновику | Галлюцинации к черновику |",
        "|---|---|---|---|---|---|---|",
    ]
    for generator, rows in data.items():
        if not isinstance(rows, dict):
            continue
        short = generator.split(" (")[0]
        for mode, name in modes.items():
            r = rows[mode]
            p, h = r["vs_vanilla"], r["hallucination_vs_vanilla"]
            paired = "–" if mode == "vanilla" else f"+{p['gained']}/−{p['lost']}, p = {p_value(p['mcnemar_p'])}"
            halluc = "–" if mode == "vanilla" else f"−{h['removed']}/+{h['added']}"
            lines.append(
                f"| {short} | {name} | {r['correct']} | {r['hallucination']} | {r['false_refusal']} "
                f"| {paired} | {halluc} |"
            )
    return "\n".join(lines)


def full_contour() -> dict[str, Any]:
    sig = json.loads((OUT / "significance.json").read_text("utf-8"))["parts"]
    a = sig["all"]
    return {
        "correct": a["draft_vs_full"]["full"],
        "v3": sig["v3"]["draft_vs_full"]["full"],
        "xquad": sig["xquad-full"]["draft_vs_full"]["full"],
        "hallucination": a["hallucination"]["full"]["k"],
        "false_refusal": a["false_refusal"]["full"],
        "gained": a["draft_vs_full"]["gained"],
        "lost": a["draft_vs_full"]["lost"],
    }


def main() -> None:
    print("Таблица 5\n" + table5(full_contour()) + "\n")
    print("Таблица 6\n" + table6() + "\n")
    print("Таблица 7\n" + table7())


if __name__ == "__main__":
    main()
