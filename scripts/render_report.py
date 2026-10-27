"""Render reports/EVALUATION_REPORT.md, CSV tables, SVG figures and reports/summary.json.

Every number is read from one run directory produced by evaluate_suite.py; nothing is typed
by hand. Sections whose mode has not been run are reported as missing.

Usage:  python scripts/render_report.py reports/runs/<run_id> [--baseline reports/runs/<old_run_id>]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_dataset import EvalDatasetV3, load_dataset  # noqa: E402
from evaluate_suite import mcnemar_exact  # noqa: E402
from scripts.svg_charts import grouped_bars, heatmap, stacked_hbars  # noqa: E402

REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MODE_NAMES = {
    "dense": "Dense (BGE-M3)",
    "sparse": "Sparse (BM25)",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_rerank_unfiltered": "RRF + rerank (без порога)",
    "hybrid_rerank": "RRF + rerank (прод)",
}
MODE_ORDER = ("dense", "sparse", "hybrid_rrf", "hybrid_rerank_unfiltered", "hybrid_rerank")
CHART_NAMES = ("Dense", "BM25", "RRF", "RRF+rerank, без порога", "RRF+rerank, прод")
PROFILE_SHORT = {"guard_on": "guard вкл.", "guard_off": "guard выкл."}
CATEGORY_NAMES = {
    "single_hop": "single_hop",
    "multi_hop": "multi_hop",
    "tabular_code": "tabular_code",
    "adversarial": "adversarial",
    "unanswerable": "unanswerable",
    "lexical_trap": "lexical_trap",
}
LABEL_RU = {"entailment": "Entailment", "neutral": "Neutral", "contradiction": "Contradiction"}
PROFILE_NAMES = {"guard_on": "premise guard вкл. (прод)", "guard_off": "premise guard выкл."}
CAUSE_NAMES = {
    "nli_verification_failed": "ответ заменён отказом по итогам NLI-проверки",
    "filtered_answer": "ответ опубликован, но фильтр удалил утверждение с ключевым фактом",
    "relevance_threshold_not_met": "отказ по порогу релевантности",
    "llm_refusal": "отказ LLM",
}


def reason_names(run: Run | None = None) -> dict[str, str]:
    """Claim status/reason labels with the thresholds recorded for the run."""

    def value(name: str, default: float) -> Any:
        recorded = run.setting(name) if run else None
        return default if recorded is None else recorded

    con = value("nli_contradiction_threshold", 0.2)
    return {
        "verified:none": "VERIFIED",
        "unverified:missing_citation": "UNVERIFIED: нет маркера [S#] (и нечего унаследовать), NLI не вызывался",
        "unverified:cited_support_not_established": f"UNVERIFIED: ни чанк, ни окно не подтвердили, P(con) < {con}",
        "contradiction:cited_support_not_established": (
            f"CONTRADICTION: P(con) ≥ {con} без подтверждающего окна"
            + (
                f" или вето P(con) ≥ {value('nli_veto_contradiction_threshold', 0.9)}"
                if run and run.setting("nli_aggregation") == "support_first"
                else ""
            )
        ),
        "unverified:anchor_not_in_source": "UNVERIFIED: число, код или идентификатор утверждения отсутствует в источнике",
        "contradiction:anchor_not_in_source": f"CONTRADICTION: якорь отсутствует в источнике и P(con) ≥ {con}",
        "unverified:absence_statement": "Оговорка об отсутствии сведений (публикуется, не считается заблокированной)",
        "unverified:verdict_marker": "Вердикт «Нет.» / «Неверно.» (публикуется вместе со следующим VERIFIED)",
    }


def support_names(run: Run) -> dict[str | None, str]:
    """How a VERIFIED claim was supported, with the recorded window thresholds."""
    return {
        "window": f"окно до {run.setting('nli_window_sentences')} предложений, P(ent) ≥ "
        f"{run.setting('nli_window_entailment_threshold')} и P(con) < "
        f"{run.setting('nli_window_contradiction_threshold')}",
        "anchor_window": "окно со всеми якорями утверждения, P(ent) ≥ "
        f"{run.setting('nli_anchor_window_entailment_threshold')} и P(con) < "
        f"{run.setting('nli_anchor_window_contradiction_threshold')}",
        "code_span": f"дословный фрагмент синтаксиса ≥ {run.setting('nli_code_span_min_chars')} символов",
        None: "целый чанк (базовое правило)",
    }


STEP_NAMES = {
    "baseline": "Базовый верификатор (bf16, целый чанк, строгое правило)",
    "nli_float16": "+ NLI в float16 с пакетной обработкой",
    "segmentation_v2": "+ сегментация v2 (сокращения, «;»)",
    "citation_inheritance": "+ наследование ссылки [S#] соседнего предложения",
    "premise_cleanup": "+ очистка посылки (заголовок, HTML, сущности)",
    "sentence_windows": "+ оконная проверка (1-2 предложения)",
    "support_first": "+ правило «поддержка важнее шума» с вето P(con) ≥ 0.9",
    "anchor_guard": "+ якоря: числа и идентификаторы обязаны быть в источнике",
    "code_span": "+ дословные фрагменты синтаксиса",
    "abstention_guard": "+ защита оговорок и исправлений предпосылки",
}
# Heuristic: a citation-less claim that is a cut piece of a sentence rather than a whole one.
FRAGMENT = re.compile(r"(?:\b(?:Rev|No|Sec|Vol|Fig|vs)\.|;)$|^\d")


# --------------------------------------------------------------------------- formatting


def pct(value: float | None, digits: int = 1) -> str:
    return "н/д" if value is None else f"{value * 100:.{digits}f} %"


def num(value: float | None, digits: int = 3) -> str:
    return "н/д" if value is None else f"{value:.{digits}f}"


def pval(value: float | None) -> str:
    if value is None:
        return "н/д"
    return f"{value:.1e}" if value < 1e-3 else f"{value:.3f}"


def direction(
    before: float | None,
    after: float | None,
    forms: tuple[str, str, str] = ("снизилась", "выросла", "не изменилась"),
) -> str:
    if before is None or after is None:
        return "н/д"
    return forms[0] if after < before else forms[1] if after > before else forms[2]


def plural(n: int, one: str, few: str, many: str) -> str:
    """Russian noun agreement: 1 утверждение, 2 утверждения, 5 утверждений."""
    tail = n % 100
    if 11 <= tail <= 14 or tail % 10 == 0 or tail % 10 >= 5:
        return f"{n} {many}"
    return f"{n} {one}" if tail % 10 == 1 else f"{n} {few}"


def count(r: dict[str, Any] | None) -> str:
    return "н/д" if not r or not r.get("n") else f"{r['k']} из {r['n']}"


def ms(value: float | None) -> str:
    return "н/д" if value is None else f"{value:,.0f}".replace(",", " ")


def rate_cell(r: dict[str, Any] | None) -> str:
    if not r or not r.get("n"):
        return "н/д"
    lo, hi = r["wilson95"]
    return f"{r['rate'] * 100:.1f} % ({r['k']}/{r['n']}; ДИ {lo * 100:.0f}-{hi * 100:.0f} %)"


def table(header: Sequence[str], rows: Iterable[Sequence[Any]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(name: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return f"tables/{name}"


def write_svg(name: str, content: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / name).write_text(content, encoding="utf-8", newline="\n")
    return f"figures/{name}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def quote(text: str | None, limit: int = 220) -> str:
    value = " ".join((text or "").split()).replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1] + "…"


# --------------------------------------------------------------------------- run data


class Run:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.rel = run_dir.relative_to(ROOT).as_posix()
        self.config = read_json(run_dir / "config.json") or {}
        self.retrieval = read_json(run_dir / "retrieval" / "summary.json")
        self.retrieval_cases = read_jsonl(run_dir / "retrieval" / "cases.jsonl")
        self.nli = read_json(run_dir / "nli" / "summary.json")
        self.nli_pairs = read_jsonl(run_dir / "nli" / "pairs.jsonl")
        self.e2e = {
            profile: read_json(run_dir / f"e2e_{profile}" / "summary.json")
            for profile in ("guard_on", "guard_off")
        }
        self.e2e_cases = {
            profile: read_jsonl(run_dir / f"e2e_{profile}" / "cases.jsonl")
            for profile in ("guard_on", "guard_off")
        }
        self.profiles = [p for p in ("guard_on", "guard_off") if self.e2e[p]]
        self.ablation = read_json(run_dir / "replay" / "ablation.json")
        self.replay = {
            profile: read_json(run_dir / "replay" / f"e2e_{profile}" / "summary.json")
            for profile in ("guard_on", "guard_off")
        }
        self.pilots = read_json(run_dir / "pilots" / "summary.json")
        # Rows of cases that failed for infrastructure reasons and were run again
        # (evaluate_suite.py e2e --retry-errors); the original error records are kept.
        self.retried = {
            profile: read_jsonl(run_dir / f"e2e_{profile}" / "retried_errors.jsonl")
            for profile in ("guard_on", "guard_off")
        }

    def setting(self, name: str) -> Any:
        """A setting of the run as recorded in config.json (first E2E profile that has it)."""
        for profile in ("guard_on", "guard_off"):
            value = (
                self.config.get("modes", {}).get(f"e2e_{profile}", {}).get("settings") or {}
            ).get(name)
            if value is not None:
                return value
        return None


def premise_lengths(dataset: EvalDatasetV3) -> dict[str, int]:
    """Premise chunk length in characters, from the local parse cache when available."""
    lengths: dict[str, int] = {}
    for path in (ROOT / ".artifacts" / "eval_v3_parse_cache").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        for chunk in record["chunks"]:
            lengths[chunk["chunk_id"]] = len(chunk["text"])
    wanted = {c.nli_probe.premise_chunk_id for c in dataset.cases}
    return {cid: n for cid, n in lengths.items() if cid in wanted}


# --------------------------------------------------------------------------- changes


def signed(before: float | None, after: float | None, scale: float = 100.0) -> str:
    """Signed difference in percentage points (scale 100) or plain units (scale 1)."""
    if before is None or after is None:
        return "н/д"
    value = (after - before) * scale
    return f"{value:+.1f}" if scale != 1.0 else f"{value:+.0f}"


def comparison_rows(run: Run, baseline: Run) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for profile in run.profiles:
        b, n = baseline.e2e.get(profile), run.e2e[profile]
        if not b:
            continue
        short = PROFILE_SHORT[profile]
        for title, variant, key in (
            ("FactRAG-NLI: корректный исход ↑", "factrag", "correct"),
            ("FactRAG-NLI: отказы на отвечаемых ↓", "factrag", "false_refusal_on_answerable"),
            ("FactRAG-NLI: галлюцинации ↓", "factrag", "hallucination"),
            ("Vanilla RAG: корректный исход", "vanilla", "correct"),
            ("Vanilla RAG: галлюцинации", "vanilla", "hallucination"),
        ):
            rows.append(
                [
                    f"{title} ({short})",
                    rate_cell(b[variant][key]),
                    rate_cell(n[variant][key]),
                    signed(b[variant][key]["rate"], n[variant][key]["rate"]),
                ]
            )
        before = len(b["hallucination_ids"]["factrag_only"])
        after = len(n["hallucination_ids"]["factrag_only"])
        rows.append(
            [
                f"Галлюцинации, внесённые фильтром ↓ ({short})",
                before,
                after,
                signed(before, after, 1.0),
            ]
        )
    if run.nli and baseline.nli:
        bd, nd = baseline.nli["overall"]["decision_rule"], run.nli["overall"]["decision_rule"]
        for title, key in (
            ("NLI-пробы: ложная отбраковка верных ↓", "false_rejection_rate"),
            ("NLI-пробы: перехват искажённых и неподтверждённых ↑", "hallucination_catch_rate"),
        ):
            rows.append([title, pct(bd[key]), pct(nd[key]), signed(bd[key], nd[key])])
        for lang in ("en", "ru"):
            b_frr = baseline.nli["by_hypothesis_language"][lang]["decision_rule"][
                "false_rejection_rate"
            ]
            n_frr = run.nli["by_hypothesis_language"][lang]["decision_rule"]["false_rejection_rate"]
            rows.append(
                [
                    f"NLI-пробы: ложная отбраковка, {lang.upper()}-гипотезы ↓",
                    pct(b_frr),
                    pct(n_frr),
                    signed(b_frr, n_frr),
                ]
            )
    if run.retrieval and baseline.retrieval:
        bo, no = baseline.retrieval, run.retrieval
        slices = [
            ("Поиск (прод): Recall@5 ↑", bo["overall"], no["overall"], "recall@5"),
            ("Поиск (прод): MRR@10 ↑", bo["overall"], no["overall"], "mrr@10"),
            (
                "Поиск: Recall@5, RU-вопросы ↑",
                bo["by_language"]["ru"],
                no["by_language"]["ru"],
                "recall@5",
            ),
        ] + [
            (f"Поиск: Recall@5, {c} ↑", bo["by_category"][c], no["by_category"][c], "recall@5")
            for c in ("tabular_code", "multi_hop")
            if c in bo["by_category"] and c in no["by_category"]
        ]
        for title, b_slice, n_slice, key in slices:
            b_val, n_val = b_slice["hybrid_rerank"][key], n_slice["hybrid_rerank"][key]
            rows.append([title, num(b_val), num(n_val), f"{n_val - b_val:+.3f}"])
        for title, key in (
            ("Порог релевантности: отсечено вопросов без ответа", "rejected_unanswerable"),
            ("Порог релевантности: отсечено отвечаемых ↓", "rejected_answerable"),
        ):
            b_gate, n_gate = bo["relevance_gate"][key], no["relevance_gate"][key]
            rows.append(
                [
                    title,
                    rate_cell(b_gate),
                    rate_cell(n_gate),
                    signed(b_gate["rate"], n_gate["rate"]),
                ]
            )
    return rows


def ablation_rows(run: Run) -> tuple[list[list[Any]], list[list[Any]]]:
    """Report rows and CSV rows of the replay ablation (guard_on drafts of the source run)."""
    rows: list[list[Any]] = []
    csv_rows: list[list[Any]] = []
    previous: int | None = None
    for step in (run.ablation or {}).get("steps", []):
        e2e = step["e2e"].get("guard_on") or next(iter(step["e2e"].values()))
        f = e2e["factrag"]
        nli = step["nli"]["overall"]
        correct = f["correct"]["k"]
        rows.append(
            [
                STEP_NAMES.get(step["step"], step["step"]),
                f"{correct} из {f['correct']['n']}",
                "" if previous is None else f"{correct - previous:+d}",
                count(f["false_refusal_on_answerable"]),
                count(f["hallucination"]),
                len(e2e["filter_induced_hallucinations"]),
                pct(nli["false_rejection_rate"]),
                pct(nli["hallucination_catch_rate"]),
            ]
        )
        csv_rows.append(
            [
                step["step"],
                *(
                    f[key]["k"]
                    for key in ("correct", "false_refusal_on_answerable", "hallucination")
                ),
                len(e2e["filter_induced_hallucinations"]),
                nli["false_rejection_rate"],
                nli["hallucination_catch_rate"],
                *(
                    e2e["by_split"].get(split, {}).get("factrag", {}).get("correct", {}).get("k")
                    for split in ("dev", "test")
                ),
            ]
        )
        previous = correct
    return rows, csv_rows


def catch_note(steps: Sequence[dict[str, Any]]) -> str:
    """Largest correctness gain and what it cost in NLI catch rate along the ladder."""
    if len(steps) < 2:
        return ""

    def correct(step: dict[str, Any]) -> int:
        e2e = step["e2e"].get("guard_on") or next(iter(step["e2e"].values()))
        return int(e2e["factrag"]["correct"]["k"])

    def catch(step: dict[str, Any]) -> float:
        return float(step["nli"]["overall"]["hallucination_catch_rate"])

    gains = [(correct(b) - correct(a), b, a) for a, b in zip(steps, steps[1:])]
    gain, best, before = max(gains, key=lambda item: item[0])
    lowest = min(steps, key=catch)
    text = (
        f"Наибольший вклад даёт шаг «{STEP_NAMES.get(best['step'], best['step'])}» "
        f"({gain:+d} корректных исходов)"
    )
    if catch(lowest) < catch(steps[0]):
        text += (
            f"; перехват NLI-проб на пути опускается до {pct(catch(lowest))} (шаг "
            f"«{STEP_NAMES.get(lowest['step'], lowest['step'])}») и в итоге составляет "
            f"{pct(catch(steps[-1]))} против {pct(catch(steps[0]))} у базового верификатора"
        )
    return text + "."


def split_rows(run: Run) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for split in ("dev", "test"):
        for profile in run.profiles:
            block = run.e2e[profile].get("by_split", {}).get(split)
            if not block:
                continue
            f, v = block["factrag"], block["vanilla"]
            rows.append(
                [
                    f"{split}, {PROFILE_SHORT[profile]}",
                    rate_cell(v["correct"]),
                    rate_cell(f["correct"]),
                    rate_cell(f["false_refusal_on_answerable"]),
                    f"{count(v['hallucination'])} → {count(f['hallucination'])}",
                ]
            )
    return rows


def section_changes(run: Run, baseline: Run | None) -> str:
    out = ["## 2. Что изменено и почему", ""]
    if baseline:
        rows = comparison_rows(run, baseline)
        write_csv(
            "baseline_comparison.csv",
            ["metric", "baseline", "current", "delta"],
            rows,
        )
        out += [
            f"Базовый прогон [`{baseline.rel}`]({baseline.dir.relative_to(REPORTS).as_posix()}) и "
            "этот прогон выполнены на одних и тех же вопросах датасета v3, корпусе, моделях и "
            "стенде. "
            "Между ними изменены три вещи: настройки гибридного поиска, системный промпт и "
            "верификатор; вклад верификатора отдельно измерен ниже повторной проверкой "
            "сохранённых черновиков базового прогона.",
            "",
            table(["Метрика", "Базовый прогон", "Этот прогон", "Δ, п.п."], rows, "lrrr"),
            "",
            "Таблица: [`tables/baseline_comparison.csv`](tables/baseline_comparison.csv); "
            "для поиска Δ дана в абсолютных единицах метрики.",
            "",
        ]
    steps, csv_rows = ablation_rows(run)
    if steps:
        write_csv(
            "replay_ablation.csv",
            [
                "step",
                "factrag_correct",
                "false_refusals_on_answerable",
                "factrag_hallucinations",
                "filter_induced_hallucinations",
                "nli_false_rejection_rate",
                "nli_catch_rate",
                "factrag_correct_dev",
                "factrag_correct_test",
            ],
            csv_rows,
        )
        first = run.ablation["steps"][0]["e2e"]
        profile = "guard_on" if "guard_on" in first else next(iter(first))
        vanilla = first[profile]["vanilla"]
        out += [
            "### 2.1. Абляция верификатора на тех же черновиках",
            "",
            f"Режим `evaluate_suite.py replay` заново проверяет каждый сохранённый черновик LLM "
            f"прогона `{run.ablation['source_run']}` (профиль {PROFILE_SHORT[profile]}) и "
            f"{run.ablation['steps'][0]['nli']['overall']['n']} NLI-проб, "
            "включая изменения верификатора по одному (сверху вниз накопительно). LLM и поиск не "
            "вызываются, поэтому разница между строками - вклад только этого изменения. "
            f"Черновик (Vanilla RAG) одинаков во всех строках: корректен в "
            f"{count(vanilla['correct'])}, галлюцинации {count(vanilla['hallucination'])}.",
            "",
            table(
                [
                    "Шаг",
                    "FactRAG корректно",
                    "Δ",
                    "Отказы на отвечаемых",
                    "Галлюцинации",
                    "Внесены фильтром",
                    "NLI: ложная отбраковка",
                    "NLI: перехват",
                ],
                steps,
                "lrrrrrrr",
            ),
            "",
            "Первая строка воспроизводит базовый прогон; последняя - текущий профиль. "
            + catch_note(run.ablation["steps"])
            + " Таблица с разбивкой dev/test: "
            "[`tables/replay_ablation.csv`](tables/replay_ablation.csv).",
            "",
        ]
    out += [
        "### 2.2. Инженерные решения",
        "",
        "- **Сегментация v2** (`grounding.segment_claims`). Прежнее разбиение резало "
        "«SP 800-53 Rev. 5», «No. 1», «e.g.» и «LIKE ... INCLUDING» на обрывки и считало «;» "
        "границей утверждения; обрывок до маркера [S#] оставался без цитаты и блокировался без "
        "вызова NLI. Теперь сокращения и номера защищены, «;» не делит утверждение, хвост "
        "ссылок вида «[S1], [S2]» остаётся с предложением.",
        "- **Наследование ссылки** (`grounding.cited_claims`). Предложение без [S#] получает "
        "ссылку следующего предложения того же абзаца, затем предыдущего, затем ближайшую в "
        "ответе; опубликованное унаследованное утверждение получает маркер явно. Неверный "
        "маркер ([S0], [S 1]) не наследуется: такое утверждение отклоняется, как и раньше.",
        "- **Очистка посылки и оконная проверка** (`NLIFactVerifier._evidence`). Длинный чанк "
        "смешивает нужное предложение с соседними строками таблиц и условиями, и mDeBERTa "
        "выдаёт ложное противоречие на дословно подтверждённом утверждении. Посылка очищается "
        "от служебного заголовка и HTML, затем одним пакетом оцениваются чанк и все окна из 1-2 "
        "предложений. Окно поддерживает утверждение при строгих порогах; окно, содержащее все "
        "числа и идентификаторы утверждения, - при мягких; дословный фрагмент синтаксиса (ABNF, "
        "флаги) - при отсутствии сильного противоречия.",
        "- **Якоря** (`grounding.claim_anchors`). Число, код, флаг или идентификатор утверждения "
        "обязан встречаться в процитированном чанке (с учётом границ числа: «20» не находится в "
        "«2020»); иначе утверждение не может быть подтверждено никакой вероятностью NLI. "
        "Обозначения документов («RFC 9110», «SP 800-63B-4») якорями не считаются.",
        "- **Поддержка важнее шума** (`nli_aggregation=support_first`). Прежнее правило "
        f"блокировало утверждение, если любой из процитированных источников давал P(con) ≥ "
        f"{run.setting('nli_contradiction_threshold')}, даже когда другой источник его дословно "
        "подтверждал. Теперь достаточно одного подтверждающего источника; другой блокирует "
        f"только явным противоречием (P(con) ≥ {run.setting('nli_veto_contradiction_threshold')}).",
        "- **Оговорки и исправления предпосылки** (`abstention_guard`). Фраза «документ не "
        "содержит сведений о ...» не подтверждается NLI по построению; раньше фильтр удалял её и "
        "публиковал оставшийся смежный факт, внося галлюцинацию. Теперь такие фразы "
        "публикуются как оговорка и не считаются заблокированными, ответ только из оговорок "
        "становится каноническим отказом, а исправление предпосылки («Нет, ...») не публикуется "
        "без подтверждённого обоснования.",
        "- **NLI в float16 пакетами.** В bf16 вероятности одной и той же пары заметно "
        "различаются между пакетом с паддингом и одиночным вызовом; float16 совпадает с float32 "
        "по метрикам при той же памяти, поэтому окна оцениваются пакетами в float16.",
        f"- **Поиск.** Итоговый балл = alpha · RRF + (1 - alpha) · rerank; при alpha "
        f"{run.setting('rerank_rrf_alpha')} порядок задаёт кросс-энкодер, RRF (k = "
        f"{run.setting('rrf_k')}) разводит близкие баллы; реранкер оценивает "
        f"{run.setting('rerank_candidates')} кандидатов"
        + (
            f" вместо {baseline.setting('rerank_candidates')} (alpha "
            f"{baseline.setting('rerank_rrf_alpha')}, k = {baseline.setting('rrf_k')} в базовом "
            "прогоне)"
            if baseline
            else ""
        )
        + ". Вес BM25 умножается на долю терминов запроса, найденных в словаре "
        "индекса: русский вопрос к англоязычному RFC почти не даёт лексического сигнала и не "
        "должен вытеснять dense-кандидатов.",
        "- **Промпт.** Добавлены требования: каждое предложение заканчивается ссылкой, составной "
        "вопрос - отдельное предложение на каждую часть, числа, коды, идентификаторы и правила "
        "ABNF переносятся дословно. Дословный перенос делает утверждение проверяемым NLI и "
        "якорями; цена - больше англоязычных цитат в русском ответе.",
        "",
        "Пороги подобраны на dev-половине датасета (`evaluate_suite.dataset_split`: внутри каждой "
        "категории кейсы упорядочены по SHA-1 идентификатора и чередуются), test-половина "
        "отчитывается отдельно как проверка на переобучение:",
        "",
    ]
    rows = split_rows(run)
    if rows:
        out += [
            table(
                [
                    "Половина, профиль",
                    "Vanilla корректно",
                    "FactRAG корректно",
                    "FactRAG отказы на отвечаемых",
                    "Галлюцинации Vanilla → FactRAG",
                ],
                rows,
                "lrrrr",
            ),
            "",
        ]
    if run.nli and run.nli.get("by_split"):
        out += [
            "NLI-пробы по половинам: "
            + "; ".join(
                f"{split} - ложная отбраковка {pct(m['false_rejection_rate'])}, перехват "
                f"{pct(m['hallucination_catch_rate'])}"
                for split, m in run.nli["by_split"].items()
            )
            + ".",
            "",
        ]
    if run.pilots:
        pilot_rows = [
            [
                p["name"],
                count(p["vanilla"]["correct"]),
                count(p["factrag"]["correct"]),
                count(p["factrag"]["false_refusal_on_answerable"]),
                f"{count(p['vanilla']['hallucination'])} → {count(p['factrag']['hallucination'])}",
                p.get("note", ""),
            ]
            for p in run.pilots["variants"]
        ]
        out += [
            "### 2.3. Пилоты на dev-половине",
            "",
            run.pilots["description"],
            "",
            table(
                [
                    "Вариант",
                    "Vanilla корректно",
                    "FactRAG корректно",
                    "Отказы на отвечаемых",
                    "Галлюцинации Vanilla → FactRAG",
                    "Примечание",
                ],
                pilot_rows,
                "lrrrrl",
            ),
            "",
        ]
    out += [
        "### 2.4. Что не сработало",
        "",
        "- Окна из трёх предложений вместо двух не добавили подтверждённых утверждений на dev.",
        "- Замена обозначений документов в гипотезе («RFC 9110» → «документ») снижала перехват "
        "искажений: часть проб проверяет именно приписывание факта документу.",
        "- Проверка утверждения без ссылки против всех источников ответа и изменение доли "
        "заблокированных утверждений, при которой ответ заменяется отказом (0.5), исход не "
        "меняли.",
        "- Правило «отказ, если удалено первое утверждение с отрицанием» увеличивало ложные "
        "отказы; оставлено только правило для исправления предпосылки без обоснования.",
        "- Повышение точности NLI (bf16 → float16/float32) само по себе ошибки не исправляет: "
        "это видно по второй строке абляции.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- sections


def section_summary(
    run: Run, dataset: EvalDatasetV3, headline: dict[str, Any], baseline: Run | None = None
) -> str:
    lines = ["## 1. Краткие выводы", ""]
    prod = run.e2e.get("guard_on")
    base = baseline.e2e.get("guard_on") if baseline else None
    if prod:
        v, f = prod["vanilla"], prod["factrag"]
        pc = prod["paired_correctness"]
        causes = pc.get("vanilla_only_correct_by_cause", {})
        before = (
            f" (базовый прогон: {pct(base['factrag']['correct']['rate'])}, отказы на отвечаемых "
            f"{pct(base['factrag']['false_refusal_on_answerable']['rate'])})"
            if base
            else ""
        )
        lines.append(
            f"- **Сквозной результат.** На {prod['cases']} вопросах FactRAG-NLI даёт корректный "
            f"исход в {pct(f['correct']['rate'])}{before}; черновик той же LLM до NLI-фильтра "
            f"(Vanilla RAG) - {pct(v['correct']['rate'])}. Только Vanilla верна в "
            f"{plural(pc['vanilla_only_correct'], 'кейсе', 'кейсах', 'кейсах')}, только "
            f"FactRAG-NLI в {pc['factrag_only_correct']} (точный тест Мак-Немара, p = "
            f"{pval(pc['mcnemar_exact_p'])})"
            + (
                ". Причины потерь: "
                + "; ".join(f"{CAUSE_NAMES.get(k, k)} - {n}" for k, n in causes.items())
                if causes
                else ""
            )
            + f". Отказы на отвечаемых вопросах: Vanilla "
            f"{pct(v['false_refusal_on_answerable']['rate'])}, FactRAG-NLI "
            f"{pct(f['false_refusal_on_answerable']['rate'])}."
        )
        guard = [
            f"{PROFILE_SHORT[p]}: Vanilla {count(run.e2e[p]['vanilla']['hallucination'])}, "
            f"FactRAG-NLI {count(run.e2e[p]['factrag']['hallucination'])}"
            + (
                f" (базовый прогон: {count(baseline.e2e[p]['factrag']['hallucination'])})"
                if baseline and baseline.e2e.get(p)
                else ""
            )
            for p in run.profiles
        ]
        introduced = {
            p: len(run.e2e[p].get("hallucination_ids", {}).get("factrag_only", []))
            for p in run.profiles
        }
        lines.append(
            "- **Галлюцинации** (вопросы с известной истиной: без ответа, с ложной предпосылкой, "
            "lexical_trap без ответа): "
            + "; ".join(guard)
            + ". Галлюцинаций, внесённых фильтром (черновик верен, опубликованный ответ нет): "
            + ", ".join(f"{PROFILE_SHORT[p]} - {n}" for p, n in introduced.items())
            + (
                " (базовый прогон: "
                + ", ".join(
                    f"{PROFILE_SHORT[p]} - "
                    f"{len(baseline.e2e[p].get('hallucination_ids', {}).get('factrag_only', []))}"
                    for p in run.profiles
                    if baseline.e2e.get(p)
                )
                + ")"
                if baseline
                else ""
            )
            + "."
        )
        if run.nli:
            d = run.nli["overall"]["decision_rule"]
            base_nli = (
                f" (базовый прогон: {pct(baseline.nli['overall']['decision_rule']['false_rejection_rate'])} и "
                f"{pct(baseline.nli['overall']['decision_rule']['hallucination_catch_rate'])})"
                if baseline and baseline.nli
                else ""
            )
            lines.append(
                f"- **NLI-модуль.** На {run.nli['overall']['pairs']} контрольных парах с эталонной "
                f"посылкой ложная отбраковка верных утверждений {pct(d['false_rejection_rate'])}, "
                f"перехват искажённых и неподтверждённых {pct(d['hallucination_catch_rate'])}"
                + base_nli
                + ". По языку гипотезы ложная отбраковка: "
                + ", ".join(
                    f"{lang.upper()} {pct(b['decision_rule']['false_rejection_rate'])}"
                    for lang, b in run.nli["by_hypothesis_language"].items()
                )
                + "."
            )
        if run.ablation:
            steps = run.ablation["steps"]
            first, last = steps[0]["e2e"], steps[-1]["e2e"]
            profile = "guard_on" if "guard_on" in first else next(iter(first))
            lines.append(
                f"- **Вклад верификатора отдельно** (повторная проверка тех же черновиков "
                f"`{run.ablation['source_run']}`, {PROFILE_NAMES[profile]}): корректный исход "
                f"{count(first[profile]['factrag']['correct'])} → "
                f"{count(last[profile]['factrag']['correct'])}, отказы на отвечаемых "
                f"{count(first[profile]['factrag']['false_refusal_on_answerable'])} → "
                f"{count(last[profile]['factrag']['false_refusal_on_answerable'])}, внесённые "
                f"фильтром галлюцинации {len(first[profile]['filter_induced_hallucinations'])} → "
                f"{len(last[profile]['filter_induced_hallucinations'])} (раздел 2.1)."
            )
    if run.retrieval:
        o = run.retrieval["overall"]
        b_o = baseline.retrieval["overall"] if baseline and baseline.retrieval else None
        line = (
            f"- **Поиск и задержка.** Гибрид с реранкингом: Recall@5 "
            f"{num(o['hybrid_rerank']['recall@5'])}, MRR@10 {num(o['hybrid_rerank']['mrr@10'])}"
            + (
                f" (базовый прогон: {num(b_o['hybrid_rerank']['recall@5'])} / "
                f"{num(b_o['hybrid_rerank']['mrr@10'])})"
                if b_o
                else ""
            )
            + f"; dense {num(o['dense']['recall@5'])} / {num(o['dense']['mrr@10'])}, BM25 "
            f"{num(o['sparse']['recall@5'])} / {num(o['sparse']['mrr@10'])}."
        )
        if prod:
            lat = prod["latency_ms"]["generated_queries"]
            line += (
                f" NLI-верификация занимает p50 {ms(lat['nli_ms']['p50'])} мс на запрос при "
                f"полном времени p50 {ms(lat['total_ms']['p50'])} мс, из которых генерация LLM "
                f"{ms(lat['generation_ms']['p50'])} мс."
            )
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def section_method(run: Run, dataset: EvalDatasetV3) -> str:
    stats: dict[str, Any] = dict(dataset.stats)
    cats = stats["by_category"]
    previous = read_json(ROOT / "data" / "extended_dataset_v2.json")
    v2_cases = len(previous["cases"]) if previous else 0
    behavior = {
        c: Counter(x.expected_behavior for x in dataset.cases if x.category == c) for c in cats
    }
    lang = {c: Counter(x.language for x in dataset.cases if x.category == c) for c in cats}
    rows = []
    for c in (
        "single_hop",
        "multi_hop",
        "tabular_code",
        "adversarial",
        "unanswerable",
        "lexical_trap",
    ):
        b = behavior[c]
        rows.append(
            [
                c,
                cats[c],
                f"{lang[c]['ru']} / {lang[c]['en']}",
                ", ".join(f"{k}: {v}" for k, v in sorted(b.items())),
            ]
        )
    rows.append(
        [
            "**Итого**",
            stats["cases"],
            f"{stats['by_language']['ru']} / {stats['by_language']['en']}",
            "",
        ]
    )
    docs = [
        [
            d.path,
            d.chunks,
            d.derived_from or "",
            sum(1 for c in dataset.cases if c.document == d.path),
        ]
        for d in dataset.documents
    ]
    cfg = run.config
    models = cfg.get("models", {})
    sampling = cfg.get("sampling", {})
    thresholds = cfg.get("nli_thresholds", {})
    llm = models.get("llm", {})
    out = [
        "## 3. Методология и датасет",
        "",
        "### 3.1. Датасет v3",
        "",
        f"`data/eval_dataset_v3.json`: {plural(stats['cases'], 'кейс', 'кейса', 'кейсов')}"
        + (
            f" (в {stats['cases'] / v2_cases:.1f}".rstrip("0").rstrip(".")
            + f" раза больше набора v2 из {plural(v2_cases, 'кейса', 'кейсов', 'кейсов')})"
            if v2_cases
            else ""
        )
        + f", {plural(stats['evidence_units'], 'единица', 'единицы', 'единиц')} эталонного "
        f"контекста, {plural(stats['multi_chunk_cases'], 'кейс', 'кейса', 'кейсов')} с "
        "доказательствами в двух и более чанках.",
        "",
        table(["Категория", "Кейсов", "RU / EN", "Ожидаемое поведение"], rows, "lrrl"),
        "",
        "Каждый кейс содержит эталонный ответ, эталонные чанки (`gold_context`: дословные цитаты, "
        "привязанные к `chunk_id` боевого парсера с токенизатором BGE-M3 и нарезкой 320/384/50), "
        "ожидаемое поведение и NLI-пробу: посылку (чанк) и гипотезу на английском и русском с "
        "ожидаемым вердиктом Entailment / Contradiction / Neutral. Скрипт "
        "`scripts/build_eval_dataset_v3.py` проверяет, что каждая цитата и посылка найдены дословно, "
        "что для multi_hop доказательства лежат в разных чанках, что термины из вопросов без ответа "
        "не встречаются нигде в корпусе, и отсекает почти-дубликаты вопросов из прежних наборов.",
        "",
        "Разметку подготовил ИИ (`review_status = ai_draft`); экспертная проверка смысла не "
        "проводилась.",
        "",
        table(["Документ", "Чанков", "Конвертирован из", "Кейсов"], docs, "lrlr"),
        "",
        "### 3.2. Стенд и воспроизводимость",
        "",
        table(
            ["Параметр", "Значение"],
            [
                ["Коммит кода", f"`{cfg.get('git', {}).get('commit', 'н/д')[:12]}`"],
                ["GPU", cfg.get("gpu") or "н/д"],
                ["Python / платформа", f"{cfg.get('python')} / {cfg.get('platform')}"],
                ["Пакеты", ", ".join(f"{k} {v}" for k, v in cfg.get("packages", {}).items())],
                [
                    "LLM",
                    f"`{Path(llm.get('path', '')).name}` ({ms((llm.get('bytes') or 0) / 2**20)} МиБ), `"
                    + " ".join((cfg.get("llama_server") or {}).get("cmdline") or ["llama-server"])
                    + "`",
                ],
                [
                    "Эмбеддинги / реранкер / NLI",
                    "BGE-M3 / bge-reranker-large / mDeBERTa-v3 (XNLI, `models/mdeberta-nli`)",
                ],
                [
                    "Сэмплирование",
                    f"temperature {sampling.get('temperature')}, top_p {sampling.get('top_p')}, seed {sampling.get('seed')}, max_tokens {sampling.get('max_tokens')}, контекст {sampling.get('context_window')}",
                ],
                [
                    "Пороги NLI",
                    f"entailment > {thresholds.get('entailment')}, contradiction ≥ {thresholds.get('contradiction')}, strong entailment ≥ {thresholds.get('strong_entailment')}",
                ],
                [
                    "Датасет",
                    f"`{cfg.get('dataset', {}).get('path')}`, SHA-256 `{cfg.get('dataset', {}).get('sha256', '')[:16]}…`",
                ],
                ["Сырые логи", f"`{run.rel}/` (config.json, cases.jsonl, summary.json по режимам)"],
            ],
            "ll",
        ),
        "",
        "### 3.3. Режимы и метрики",
        "",
        "- **Поиск** (без LLM): для каждого вопроса одним вызовом получены пять ранжирований: "
        "dense (BGE-M3, Qdrant), sparse (BM25), гибрид RRF, RRF с реранкингом до порога и боевой "
        "список после порога релевантности 0.05. Recall@K считается по единицам эталонного "
        "контекста (единица найдена, если в топ-K есть любой её чанк), MRR@10 и nDCG@K с "
        "бинарной релевантностью.",
        "- **NLI** (без LLM): боевая функция `NLIFactVerifier._claim` на эталонной посылке и "
        "гипотезе. Отдельно оценены argmax трёх классов (матрица ошибок, P/R/F1) и решение "
        "боевого правила: *Hallucination Catch Rate* (доля проб Contradiction/Neutral, не "
        "получивших VERIFIED) и *False Rejection Rate* (доля проб Entailment, не получивших VERIFIED).",
        "- **Сквозной прогон**: полный конвейер на 220 вопросах. *Vanilla RAG* = черновик LLM до "
        "NLI-фильтра (`raw_answer`), *FactRAG-NLI* = опубликованный ответ; оба из одного вызова "
        "LLM, поэтому сравнение парное. Отказ = канонический текст отказа; воздержание = отказ "
        "или явная фраза об отсутствии сведений. Корректность: для отвечаемых вопросов все "
        "ключевые факты (`answer_must_contain`) присутствуют; для вопросов без ответа воздержание; "
        "для adversarial воздержание или верный факт-исправление. Галлюцинация определена только "
        "там, где истина зафиксирована датасетом: ответ на вопрос без ответа или adversarial-ответ "
        "без верного факта.",
        "",
    ]
    return "\n".join(out)


def section_retrieval(run: Run) -> str:
    s = run.retrieval
    if not s:
        return "### 4.1. Поиск\n\nРежим не запускался.\n"
    o = s["overall"]
    rows = [
        [
            MODE_NAMES[m],
            num(o[m]["recall@1"]),
            num(o[m]["recall@3"]),
            num(o[m]["recall@5"]),
            num(o[m]["recall@10"]),
            num(o[m]["ndcg@5"]),
            num(o[m]["mrr@10"]),
        ]
        for m in MODE_ORDER
    ]
    csv_rel = write_csv(
        "retrieval_overall.csv",
        [
            "mode",
            "recall@1",
            "recall@3",
            "recall@5",
            "recall@10",
            "hit@5",
            "ndcg@5",
            "ndcg@10",
            "mrr@10",
        ],
        [
            [
                m,
                *(
                    o[m][k]
                    for k in (
                        "recall@1",
                        "recall@3",
                        "recall@5",
                        "recall@10",
                        "hit@5",
                        "ndcg@5",
                        "ndcg@10",
                        "mrr@10",
                    )
                ),
            ]
            for m in MODE_ORDER
        ],
    )
    fig = write_svg(
        "retrieval_recall.svg",
        grouped_bars(
            "Recall@K по режимам поиска (172 вопроса с эталонным контекстом)",
            list(CHART_NAMES),
            ["Recall@1", "Recall@3", "Recall@5"],
            [[o[m]["recall@1"], o[m]["recall@3"], o[m]["recall@5"]] for m in MODE_ORDER],
            ordinal=True,
            width=820,
        ),
    )
    cats = s["by_category"]
    cat_rows = [[c, *(num(cats[c][m]["recall@5"], 2) for m in MODE_ORDER)] for c in cats]
    langs = s["by_language"]
    lang_rows = [
        [lang.upper(), *(num(langs[lang][m]["recall@5"], 2) for m in MODE_ORDER)] for lang in langs
    ]
    write_csv(
        "retrieval_by_category.csv",
        ["category", *MODE_ORDER],
        [[c, *(cats[c][m]["recall@5"] for m in MODE_ORDER)] for c in cats]
        + [[f"lang_{lang}", *(langs[lang][m]["recall@5"] for m in MODE_ORDER)] for lang in langs],
    )
    gate = s["relevance_gate"]
    return "\n".join(
        [
            "### 4.1. Поиск: абляция режимов",
            "",
            table(
                ["Режим", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "nDCG@5", "MRR@10"], rows
            ),
            "",
            f"![Recall@K по режимам поиска]({fig})",
            "",
            f"Таблица: [`{csv_rel}`]({csv_rel}). Recall@5 по категориям и языку вопроса:",
            "",
            table(["Срез", *(MODE_NAMES[m] for m in MODE_ORDER)], cat_rows + lang_rows),
            "",
            f"**Порог релевантности (фильтр 1).** Отсечено {rate_cell(gate['rejected_unanswerable'])} "
            f"вопросов без ответа и {rate_cell(gate['rejected_answerable'])} отвечаемых вопросов "
            f"(по ним LLM не вызывается). Задержка поиска p50 {ms(s['latency_ms']['search']['p50'])} мс, "
            f"реранкинга p50 {ms(s['latency_ms']['rerank']['p50'])} мс.",
            "",
        ]
    )


def section_nli(run: Run) -> str:
    s = run.nli
    if not s:
        return "### 4.2. NLI-верификация\n\nРежим не запускался.\n"
    blocks = {
        "Все": s["overall"],
        "EN-гипотеза": s["by_hypothesis_language"]["en"],
        "RU-гипотеза": s["by_hypothesis_language"]["ru"],
    }
    rows = []
    for name, b in blocks.items():
        pc = b["argmax"]["per_class"]
        rows.append(
            [
                name,
                *(
                    f"{num(pc[lab]['precision'], 2)} / {num(pc[lab]['recall'], 2)} / {num(pc[lab]['f1'], 2)}"
                    for lab in ("entailment", "neutral", "contradiction")
                ),
                num(b["argmax"]["macro_f1"]),
                num(b["argmax"]["accuracy"]),
            ]
        )
    write_csv(
        "nli_per_class.csv",
        ["slice", "label", "precision", "recall", "f1", "support"],
        [
            [
                name,
                lab,
                *(
                    b["argmax"]["per_class"][lab][k]
                    for k in ("precision", "recall", "f1", "support")
                ),
            ]
            for name, b in blocks.items()
            for lab in ("entailment", "neutral", "contradiction")
        ],
    )
    dec_rows = []
    for name, b in list(blocks.items()) + [(c, s["by_category"][c]) for c in s["by_category"]]:
        d = b["decision_rule"]
        dec_rows.append(
            [
                name,
                b["pairs"],
                pct(d["hallucination_catch_rate"]),
                pct(d["contradiction_catch_rate"]),
                pct(d["neutral_catch_rate"]),
                pct(d["false_rejection_rate"]),
            ]
        )
    write_csv(
        "nli_decision_rule.csv",
        [
            "slice",
            "pairs",
            "hallucination_catch_rate",
            "contradiction_catch_rate",
            "neutral_catch_rate",
            "false_rejection_rate",
        ],
        [
            [
                r[0],
                r[1],
                *(
                    b["decision_rule"][k]
                    for k in (
                        "hallucination_catch_rate",
                        "contradiction_catch_rate",
                        "neutral_catch_rate",
                        "false_rejection_rate",
                    )
                ),
            ]
            for r, b in zip(
                dec_rows,
                list(blocks.values()) + [s["by_category"][c] for c in s["by_category"]],
                strict=True,
            )
        ],
    )
    labels = ("entailment", "neutral", "contradiction")
    figs = []
    for key, title in (("en", "EN-гипотезы"), ("ru", "RU-гипотезы")):
        cm = s["by_hypothesis_language"][key]["argmax"]["confusion_matrix"]
        figs.append(
            write_svg(
                f"nli_confusion_{key}.svg",
                heatmap(
                    f"Матрица ошибок mDeBERTa (argmax), {title}",
                    [LABEL_RU[lab] for lab in labels],
                    [LABEL_RU[lab] for lab in labels],
                    [[cm[g][p] for p in labels] for g in labels],
                ),
            )
        )
    status = s["overall"]["decision_rule"]["status_matrix"]
    status_rows = [
        [LABEL_RU[g], *(status[g].get(st, 0) for st in ("verified", "unverified", "contradiction"))]
        for g in labels
    ]
    lat = s["latency_ms_per_pair"]
    con = run.setting("nli_contradiction_threshold") or 0.2
    windows = bool(run.setting("nli_window_scoring"))
    if windows:
        rule = (
            "**Боевое правило.** Утверждение VERIFIED, если его поддерживает окно посылки "
            "(маршруты ниже) или весь чанк (P(ent) > "
            f"{run.setting('nli_entailment_threshold')} при P(con) < {con}, либо P(ent) ≥ "
            f"{run.setting('nli_strong_entailment_threshold')}), и все числа и идентификаторы "
            f"утверждения есть в источнике; CONTRADICTION, если поддержки нет и P(con) ≥ {con}. "
            "Вероятности в таблицах ниже - решающего свидетельства: окна, если оно поддержало "
            "утверждение, иначе чанка."
        )
        names = support_names(run)
        route_rows = [
            [
                names.get(route, str(route)),
                *(
                    sum(
                        p["status"] == "verified" and p.get("support") == route and p["label"] == g
                        for p in run.nli_pairs
                    )
                    for g in labels
                ),
            ]
            for route in ("window", "anchor_window", "code_span", None)
        ]
        detail = [
            "Чем подтверждены VERIFIED-вердикты (строка = маршрут, столбец = эталон пробы; "
            "VERIFIED на Neutral/Contradiction - пропущенные ошибки):",
            "",
            table(["Маршрут поддержки", *(LABEL_RU[g] for g in labels)], route_rows),
            "",
        ]
    else:
        rule = (
            f"**Боевое правило (P(con) ≥ {con} в любом источнике → CONTRADICTION; иначе "
            f"VERIFIED при P(ent) > {run.setting('nli_entailment_threshold')} во всех источниках "
            f"или ≥ {run.setting('nli_strong_entailment_threshold')} в одном):**"
        )
        sweep = s["threshold_sweep"]
        write_csv(
            "nli_threshold_sweep.csv",
            [
                "entailment_threshold",
                "contradiction_threshold",
                "hallucination_catch_rate",
                "false_rejection_rate",
            ],
            [
                [
                    r["entailment_threshold"],
                    r["contradiction_threshold"],
                    r["hallucination_catch_rate"],
                    r["false_rejection_rate"],
                ]
                for r in sweep
            ],
        )
        detail = [
            "**Чувствительность к порогу противоречия** (порог entailment 0.4, strong 0.7):",
            "",
            table(
                ["Порог contradiction", "Catch rate", "False rejection"],
                [
                    [
                        f"{r['contradiction_threshold']:.1f}"
                        + (" (прод)" if r["contradiction_threshold"] == con else ""),
                        pct(r["hallucination_catch_rate"]),
                        pct(r["false_rejection_rate"]),
                    ]
                    for r in sweep
                    if r["entailment_threshold"] == 0.4
                ],
            ),
            "",
            "Полная сетка: [`tables/nli_threshold_sweep.csv`](tables/nli_threshold_sweep.csv).",
            "",
        ]
    exceeding = s["overall"]["pairs_exceeding_context"]
    return "\n".join(
        [
            "### 4.2. NLI-верификация на контрольных парах",
            "",
            f"{s['overall']['pairs']} пар ({s['overall']['pairs'] // 2} проб × EN/RU); "
            + (
                "ни одна не превысила контекст NLI 512 токенов."
                if not exceeding
                else f"контекст NLI 512 токенов превысили {exceeding}."
            ),
            "",
            "**Классификатор (argmax трёх классов), P / R / F1:**",
            "",
            table(["Срез", "Entailment", "Neutral", "Contradiction", "Macro-F1", "Accuracy"], rows),
            "",
            f"![Матрица ошибок, EN]({figs[0]}) ![Матрица ошибок, RU]({figs[1]})",
            "",
            rule,
            "",
            table(
                ["Срез", "Пар", "Catch rate", "Contradiction", "Neutral", "False rejection"],
                dec_rows,
            ),
            "",
            "Распределение вердиктов правила (строка = эталон):",
            "",
            table(["Эталон", "VERIFIED", "UNVERIFIED", "CONTRADICTION"], status_rows),
            "",
            *detail,
            f"Время одной пробы{' (чанк и все окна одним пакетом)' if windows else ''}: p50 "
            f"{num(lat['p50'], 1)} мс, p95 {num(lat['p95'], 1)} мс.",
            "",
        ]
    )


def section_e2e(run: Run) -> str:
    if not run.profiles:
        return "### 4.3. Сквозной прогон\n\nРежим не запускался.\n"
    rows, csv_rows = [], []
    for profile in run.profiles:
        s = run.e2e[profile]
        for variant, title in (("vanilla", "Vanilla RAG"), ("factrag", "FactRAG-NLI")):
            v = s[variant]
            rows.append(
                [
                    f"{title}, {PROFILE_NAMES[profile]}",
                    rate_cell(v["hallucination"]),
                    rate_cell(v["false_refusal_on_answerable"]),
                    rate_cell(v["key_fact_accuracy_on_answerable"]),
                    rate_cell(v["correct"]),
                ]
            )
            csv_rows.append(
                [
                    profile,
                    variant,
                    *(
                        v[k]["rate"]
                        for k in (
                            "hallucination",
                            "false_refusal_on_answerable",
                            "key_fact_accuracy_on_answerable",
                            "correct",
                        )
                    ),
                ]
            )
    write_csv(
        "e2e_variants.csv",
        [
            "profile",
            "variant",
            "hallucination_rate",
            "false_refusal_rate",
            "key_fact_accuracy",
            "correct_rate",
        ],
        csv_rows,
    )
    fig = write_svg(
        "e2e_hallucination.svg",
        grouped_bars(
            "Vanilla RAG и FactRAG-NLI: корректный исход, галлюцинации, отказы на отвечаемых",
            [
                f"{metric}, {PROFILE_SHORT[p]}"
                for metric in ("Корректно", "Галлюцинации", "Отказы")
                for p in run.profiles
            ],
            ["Vanilla RAG", "FactRAG-NLI"],
            [
                [run.e2e[p]["vanilla"][key]["rate"], run.e2e[p]["factrag"][key]["rate"]]
                for key in ("correct", "hallucination", "false_refusal_on_answerable")
                for p in run.profiles
            ],
            width=1000,
        ),
    )
    paired = []
    for profile in run.profiles:
        s = run.e2e[profile]
        pc = s["paired_correctness"]
        causes = pc.get("vanilla_only_correct_by_cause", {})
        ids = s.get("hallucination_ids", {})
        paired.append(
            f"- {PROFILE_NAMES[profile]}: только Vanilla корректна в "
            f"{plural(pc['vanilla_only_correct'], 'кейсе', 'кейсах', 'кейсах')} ("
            + ", ".join(f"{CAUSE_NAMES.get(k, k)}: {n}" for k, n in causes.items())
            + f"), только FactRAG-NLI в {pc['factrag_only_correct']}; точный тест Мак-Немара p = "
            f"{pval(pc['mcnemar_exact_p'])}. Галлюцинации: Vanilla {count(s['vanilla']['hallucination'])}, "
            f"FactRAG-NLI {count(s['factrag']['hallucination'])}; только у Vanilla: "
            f"{', '.join(ids.get('vanilla_only', [])) or 'нет'}, только у FactRAG-NLI: "
            f"{', '.join(ids.get('factrag_only', [])) or 'нет'}, у обоих: "
            f"{', '.join(ids.get('both', [])) or 'нет'}."
        )
    if len(run.profiles) == 2:
        on = {r["id"]: r for r in run.e2e_cases["guard_on"] if r.get("ok")}
        off = {r["id"]: r for r in run.e2e_cases["guard_off"] if r.get("ok")}
        common = sorted(set(on) & set(off))
        for variant, title in (("vanilla", "Vanilla"), ("factrag", "FactRAG-NLI")):
            b = sum(on[i][variant]["correct"] and not off[i][variant]["correct"] for i in common)
            c = sum(off[i][variant]["correct"] and not on[i][variant]["correct"] for i in common)
            paired.append(
                f"- Premise guard (вкл. против выкл.), {title}: лучше с guard в "
                f"{plural(b, 'кейсе', 'кейсах', 'кейсах')}, без guard в {c}; точный тест "
                f"Мак-Немара p = {pval(mcnemar_exact(b, c))}."
            )
    cat_rows = []
    profile = "guard_on" if "guard_on" in run.profiles else run.profiles[0]
    s = run.e2e[profile]
    for c in (
        "single_hop",
        "multi_hop",
        "tabular_code",
        "adversarial",
        "unanswerable",
        "lexical_trap",
    ):
        v, f = s["vanilla"]["by_category"].get(c), s["factrag"]["by_category"].get(c)
        if not v:
            continue
        cat_rows.append(
            [
                c,
                f"{pct(v['correct']['rate'], 0)} → {pct(f['correct']['rate'], 0)}",
                f"{pct(v['hallucination']['rate'], 0)} → {pct(f['hallucination']['rate'], 0)}"
                if v["hallucination"]["n"]
                else "не опр.",
                f"{pct(v['refused']['rate'], 0)} → {pct(f['refused']['rate'], 0)}",
            ]
        )
    write_csv(
        "e2e_by_category.csv",
        ["profile", "category", "variant", "correct_rate", "hallucination_rate", "refusal_rate"],
        [
            [
                p,
                c,
                var,
                run.e2e[p][var]["by_category"][c]["correct"]["rate"],
                run.e2e[p][var]["by_category"][c]["hallucination"]["rate"],
                run.e2e[p][var]["by_category"][c]["refused"]["rate"],
            ]
            for p in run.profiles
            for var in ("vanilla", "factrag")
            for c in run.e2e[p][var]["by_category"]
        ],
    )
    claims = s["claims"]
    reasons = ", ".join(
        f"{k}: {v}" for k, v in sorted(s["refusal_reasons"].items(), key=lambda kv: -kv[1])
    )
    keys = list(reason_names(run)) + sorted(
        {k for p in run.profiles for k in run.e2e[p]["claims"].get("by_status_reason", {})}
        - set(reason_names(run))
    )
    reason_rows = [
        [
            reason_names(run).get(k, k),
            *(
                f"{run.e2e[p]['claims'].get('by_status_reason', {}).get(k, 0)} "
                f"({pct(run.e2e[p]['claims'].get('by_status_reason', {}).get(k, 0) / (run.e2e[p]['claims']['total'] or 1))})"
                for p in run.profiles
            ),
        ]
        for k in keys
    ]
    write_csv(
        "e2e_claim_reasons.csv",
        ["profile", "status_reason", "claims"],
        [
            [p, k, n]
            for p in run.profiles
            for k, n in run.e2e[p]["claims"].get("by_status_reason", {}).items()
        ],
    )
    return "\n".join(
        [
            "### 4.3. Сквозной прогон: Vanilla RAG и FactRAG-NLI",
            "",
            table(
                [
                    "Вариант",
                    "Галлюцинации",
                    "Отказы на отвечаемых",
                    "Ключевые факты верны",
                    "Корректный исход",
                ],
                rows,
                "lrrrr",
            ),
            "",
            f"![Корректный исход, галлюцинации и отказы]({fig})",
            "",
            *paired,
            *(
                f"- Повторно выполнены после сбоя ({PROFILE_NAMES[p]}): "
                + "; ".join(
                    f"{r['id']} - `{quote(r.get('error'), 90)}`, исходное время "
                    f"{ms(r.get('wall_ms'))} мс"
                    for r in run.retried[p]
                )
                + f". Исходные записи: `{run.rel}/e2e_{p}/retried_errors.jsonl`; в таблицах учтён "
                "результат повтора."
                for p in run.profiles
                if run.retried.get(p)
            ),
            "",
            f"По категориям ({PROFILE_NAMES[profile]}), Vanilla → FactRAG-NLI:",
            "",
            table(["Категория", "Корректный исход", "Галлюцинации", "Отказы"], cat_rows),
            "",
            "Вердикты по утверждениям черновиков (доля от всех утверждений профиля):",
            "",
            table(
                ["Статус и причина", *(PROFILE_NAMES[p] for p in run.profiles)],
                reason_rows,
                "l" + "r" * len(run.profiles),
            ),
            "",
            f"Faithfulness micro (доля VERIFIED, {PROFILE_NAMES[profile]}): "
            f"{pct(claims['faithfulness_micro'])}; пар «утверждение × источник», не поместившихся в "
            f"контекст NLI 512 токенов: {claims.get('pairs_exceeding_nli_context', 'н/д')}. "
            f"Причины отказов FactRAG-NLI: {reasons}. Эталонный чанк среди источников ответа: "
            f"{rate_cell(s['gold_in_sources'])}. Таблица: "
            "[`tables/e2e_claim_reasons.csv`](tables/e2e_claim_reasons.csv).",
            "",
        ]
    )


def section_errors(run: Run, dataset: EvalDatasetV3, lengths: dict[str, int]) -> str:
    by_id = {c.id: c for c in dataset.cases}
    out = ["## 5. Разбор ошибок", ""]
    pairs = run.nli_pairs
    if pairs:
        false_con = sorted(
            [p for p in pairs if p["label"] == "entailment" and p["status"] == "contradiction"],
            key=lambda p: -p["p_contradiction"],
        )
        missed = [p for p in pairs if p["label"] == "contradiction" and p["status"] == "verified"]
        neutral_pass = sorted(
            [p for p in pairs if p["label"] == "neutral" and p["status"] == "verified"],
            key=lambda p: -p["p_entailment"],
        )
        borderline = sorted(
            [p for p in pairs if p["label"] == "entailment" and p["status"] == "unverified"],
            key=lambda p: -p["p_entailment"],
        )
        out += [
            "### 5.1. NLI: где mDeBERTa-v3 ошибается",
            "",
            f"**Ложные противоречия** ({len(false_con)} из "
            f"{sum(p['label'] == 'entailment' for p in pairs)} проб Entailment): верное утверждение "
            f"получает CONTRADICTION и блокируется. По категориям: "
            + ", ".join(
                f"{k}: {v}" for k, v in Counter(p["category"] for p in false_con).most_common()
            )
            + "; по языку гипотезы: "
            + ", ".join(
                f"{k.upper()}: {v}"
                for k, v in sorted(Counter(p["hypothesis_language"] for p in false_con).items())
            )
            + ".",
            "",
            table(
                ["Пара", "Гипотеза", "P(ent)", "P(con)", "Посылка, симв."],
                [
                    [
                        p["id"],
                        quote(p["hypothesis"], 120),
                        num(p["p_entailment"], 3),
                        num(p["p_contradiction"], 3),
                        lengths.get(by_id[p["case_id"]].nli_probe.premise_chunk_id, "н/д"),
                    ]
                    for p in false_con[:8]
                ],
                "llrrr",
            ),
            "",
        ]
        if lengths:
            ent = [p for p in pairs if p["label"] == "entailment"]
            buckets = [(0, 800), (800, 1400), (1400, 10**9)]
            rows = []
            for lo, hi in buckets:
                sub = [
                    p
                    for p in ent
                    if lo <= lengths.get(by_id[p["case_id"]].nli_probe.premise_chunk_id, -1) < hi
                ]
                if sub:
                    rows.append(
                        [
                            f"{lo}-{hi if hi < 10**9 else '∞'}",
                            len(sub),
                            pct(sum(p["status"] == "contradiction" for p in sub) / len(sub)),
                            pct(sum(p["status"] != "verified" for p in sub) / len(sub)),
                        ]
                    )
            write_csv(
                "nli_false_rejection_by_premise_length.csv",
                ["premise_chars", "pairs", "false_contradiction_rate", "false_rejection_rate"],
                rows,
            )
            out += [
                "Ложная отбраковка верных утверждений в зависимости от длины посылки (символов в чанке):",
                "",
                table(
                    ["Длина посылки", "Пар Entailment", "Ложные противоречия", "Не VERIFIED"], rows
                ),
                "",
            ]
        out += [
            f"**Пропущенные противоречия** ({len(missed)}): искажённое утверждение получает VERIFIED. "
            "Типы искажений: "
            + ", ".join(f"{k}: {v}" for k, v in Counter(p["subtype"] for p in missed).most_common())
            + ".",
            "",
            table(
                ["Пара", "Тип искажения", "Гипотеза", "P(ent)"],
                [
                    [p["id"], p["subtype"], quote(p["hypothesis"], 110), num(p["p_entailment"], 3)]
                    for p in missed
                ],
                "lllr",
            ),
            "",
            f"**Neutral, принятые за Entailment** ({len(neutral_pass)}): утверждение, которого нет в "
            "корпусе, проходит проверку против тематически близкой посылки. По категориям: "
            + ", ".join(
                f"{k}: {v}" for k, v in Counter(p["category"] for p in neutral_pass).most_common()
            )
            + ".",
            "",
            table(
                ["Пара", "Гипотеза", "P(ent)"],
                [
                    [p["id"], quote(p["hypothesis"], 120), num(p["p_entailment"], 3)]
                    for p in neutral_pass[:8]
                ],
                "llr",
            ),
            "",
            f"**Пограничные Neutral / Entailment** ({len(borderline)}): верное утверждение, где модель "
            "не подтверждена ни окном, ни чанком, но и противоречия нет (UNVERIFIED). По категориям: "
            + ", ".join(
                f"{k}: {v}" for k, v in Counter(p["category"] for p in borderline).most_common()
            )
            + ".",
            "",
            table(
                ["Пара", "Гипотеза", "P(ent)", "P(neu)"],
                [
                    [
                        p["id"],
                        quote(p["hypothesis"], 120),
                        num(p["p_entailment"], 3),
                        num(p["p_neutral"], 3),
                    ]
                    for p in borderline[:6]
                ],
                "llrr",
            ),
            "",
        ]
    profile = (
        "guard_on" if "guard_on" in run.profiles else (run.profiles[0] if run.profiles else None)
    )
    if profile:
        cases = [r for r in run.e2e_cases[profile] if r.get("ok")]
        caught = [
            r for r in cases if r["vanilla"]["hallucination"] and not r["factrag"]["hallucination"]
        ]
        missed_e2e = [r for r in cases if r["factrag"]["hallucination"]]
        over = [
            r
            for r in cases
            if r["expected_behavior"] == "answer"
            and r["vanilla"]["correct"]
            and not r["factrag"]["correct"]
        ]

        def claim_line(r: dict[str, Any]) -> str:
            parts = []
            for claim in r["claims"][:3]:
                probs = [p for p in claim["probabilities"] if p]
                detail = (
                    f"P(ent) {num(probs[0]['p_entailment'], 2)}, P(con) {num(probs[0]['p_contradiction'], 2)}"
                    if probs
                    else claim.get("reason") or "без цитаты"
                )
                parts.append(f"{claim['status']} ({detail})")
            return "; ".join(parts) or "нет утверждений"

        over_causes = Counter(
            "filtered_answer" if not r["refused"] else r["refusal_reason"] for r in over
        )
        over_reasons = Counter(
            f"{claim['status']}:{claim.get('reason') or 'none'}"
            for r in over
            for claim in r["claims"]
        )
        uncited = [
            claim["claim"]
            for r in cases
            for claim in r["claims"]
            if claim.get("reason") == "missing_citation"
        ]
        fragments = [c for c in uncited if FRAGMENT.search(c.strip())]
        out += [
            f"### 5.2. Сквозной прогон ({PROFILE_NAMES[profile]})",
            "",
            f"**Перехваченные галлюцинации** ({len(caught)}): черновик LLM отвечает на вопрос без "
            "ответа или принимает ложную предпосылку, NLI-фильтр заменяет ответ отказом или удаляет "
            "утверждения.",
            "",
            table(
                ["Кейс", "Вопрос", "Черновик LLM (Vanilla)", "Вердикты NLI"],
                [
                    [
                        r["id"],
                        quote(by_id[r["id"]].question, 90),
                        quote(r["raw_answer"], 120),
                        claim_line(r),
                    ]
                    for r in caught[:6]
                ],
                "llll",
            ),
            "",
            f"**Галлюцинации FactRAG-NLI** ({len(missed_e2e)}): опубликован ответ на вопрос без "
            "ответа или ответ без исправления ложной предпосылки. "
            + (
                "Если черновик тоже ошибочен, NLI подтвердил утверждение, которое верно по "
                "источнику, но не отвечает на вопрос (смежный факт) или повторяет число из "
                "вопроса, встречающееся в источнике в другом месте: проверка выводимости не "
                "оценивает, отвечает ли факт на вопрос. Если черновик был верным, ошибку внёс "
                "фильтр."
                if run.setting("abstention_guard")
                else "Если черновик был верным, галлюцинацию внёс фильтр: он публикует только "
                "VERIFIED-утверждения, а фраза об отсутствии сведений или исправление "
                "предпосылки опорой в источнике не подтверждается."
            ),
            "",
            table(
                [
                    "Кейс",
                    "Вопрос",
                    "Черновик LLM (Vanilla)",
                    "Опубликованный ответ",
                    "Черновик верен",
                ],
                [
                    [
                        r["id"],
                        quote(by_id[r["id"]].question, 80),
                        quote(r["raw_answer"], 150),
                        quote(r["answer"], 110),
                        "нет" if r["vanilla"]["hallucination"] else "да",
                    ]
                    for r in missed_e2e[:6]
                ],
                "lllll",
            ),
            "",
            f"**Потерянные верные ответы** ({len(over)} из "
            f"{run.e2e[profile]['paired_correctness']['vanilla_only_correct']} исходов, которые "
            "фильтр сделал неверными; остальные - галлюцинации, внесённые фильтром, см. выше): "
            "черновик на отвечаемый вопрос содержал все ключевые факты, опубликованный ответ их "
            "не содержит. Исход: "
            + ", ".join(f"{CAUSE_NAMES.get(k, k)} - {v}" for k, v in over_causes.most_common())
            + ". Вердикты утверждений в этих черновиках: "
            + ", ".join(
                f"{reason_names(run).get(k, k)} - {v}" for k, v in over_reasons.most_common()
            )
            + ".",
            "",
            table(
                ["Кейс", "Черновик LLM", "Вердикты NLI (первые три утверждения)"],
                [[r["id"], quote(r["raw_answer"], 130), claim_line(r)] for r in over[:6]],
                "lll",
            ),
            "",
            segmentation_note(run, profile, uncited, fragments, cases),
            "",
        ]
    return "\n".join(out)


def segmentation_note(
    run: Run,
    profile: str,
    uncited: list[str],
    fragments: list[str],
    cases: list[dict[str, Any]],
) -> str:
    """Claims left without a source marker, explained for the segmentation the run used."""
    head = (
        f"**Разбиение на утверждения.** "
        f"{plural(len(uncited), 'утверждение', 'утверждения', 'утверждений')} остались без "
        "маркера [S#]"
    )
    if run.setting("claim_segmentation") != "v2":
        return (
            head + f". Из них {len(fragments)} - обрывки предложений (эвристика: оканчиваются на "
            "«Rev.», «No.», «;» или начинаются с цифры), например: "
            + ", ".join(f"«{quote(c, 60)}»" for c in list(dict.fromkeys(fragments))[:3])
            + ". Сегментация v1 защищает только русские сокращения, поэтому «SP 800-53 Rev. 5» "
            "режется после «Rev.», а «;» делит предложение до маркера источника. Остальные "
            f"{len(uncited) - len(fragments)} - целые предложения без маркера."
        )
    inherited = sum(bool(c.get("citation_inherited")) for r in cases for c in r["claims"])
    verified = sum(
        bool(c.get("citation_inherited")) and c["status"] == "verified"
        for r in cases
        for c in r["claims"]
    )
    return (
        head
        + " (сегментация v2 защищает сокращения и номера, «;» не делит утверждение); "
        + (
            f"обрывков по прежней эвристике: {len(fragments)}. "
            if fragments
            else "обрывков по прежней эвристике нет. "
        )
        + f"Ещё {plural(inherited, 'утверждение', 'утверждения', 'утверждений')} без "
        f"собственного маркера унаследовали ссылку соседнего предложения, из них VERIFIED - "
        f"{verified} ({PROFILE_NAMES[profile]})."
    )


def section_latency(run: Run, baseline: Run | None = None) -> str:
    if not run.profiles:
        return ""
    stages = ("search_ms", "rerank_ms", "generation_ms", "nli_ms")
    names = {
        "search_ms": "Поиск",
        "rerank_ms": "Реранкинг",
        "generation_ms": "Генерация LLM",
        "nli_ms": "NLI-верификация",
        "total_ms": "Итого",
        "llm_ttft_ms": "TTFT LLM",
    }
    rows, chart_rows, chart_values, csv_rows = [], [], [], []
    for profile in run.profiles:
        lat = run.e2e[profile]["latency_ms"]["generated_queries"]
        for stage in (*stages, "llm_ttft_ms", "total_ms"):
            st = lat[stage]
            if not st["n"]:
                continue
            rows.append(
                [
                    PROFILE_NAMES[profile],
                    names[stage],
                    st["n"],
                    ms(st["mean"]),
                    ms(st["p50"]),
                    ms(st["p95"]),
                ]
            )
            csv_rows.append([profile, stage, st["n"], st["mean"], st["p50"], st["p95"]])
        chart_rows.append(f"p50, {PROFILE_NAMES[profile]}")
        chart_values.append([lat[s]["p50"] or 0.0 for s in stages])
    write_csv("latency.csv", ["profile", "stage", "n", "mean_ms", "p50_ms", "p95_ms"], csv_rows)
    streaming = {
        p: (run.config.get("modes", {}).get(f"e2e_{p}", {}).get("settings") or {}).get(
            "llm_api_stream"
        )
        for p in run.profiles
    }
    ttft_note = (
        [
            "TTFT LLM не измерялся: в оценённом профиле `RAG_LLM_API_STREAM=false`, ответ "
            "llama-server приходит целиком.",
            "",
        ]
        if not any(
            run.e2e[p]["latency_ms"]["generated_queries"]["llm_ttft_ms"]["n"] for p in run.profiles
        )
        and not any(streaming.values())
        else []
    )
    fig = write_svg(
        "latency_breakdown.svg",
        stacked_hbars(
            "Медианы этапов на запрос с генерацией (подпись: сумма медиан)",
            chart_rows,
            [names[s] for s in stages],
            chart_values,
        ),
    )
    res_rows = []
    sources = [("Поиск (абляция)", run.retrieval), ("NLI-пробы", run.nli)] + [
        (f"E2E, {PROFILE_NAMES[p]}", run.e2e[p]) for p in run.profiles
    ]
    for name, summary in sources:
        r = (summary or {}).get("resources") or {}
        uses_llm = name.startswith("E2E")
        res_rows.append(
            [
                name,
                ms((r.get("gpu_used_mib") or {}).get("peak")),
                ms(r.get("torch_max_allocated_mib")),
                ms((r.get("process_rss_mib") or {}).get("peak")),
                ms((r.get("llama_server_rss_mib") or {}).get("peak"))
                if uses_llm
                else "не участвует",
            ]
        )
    init = run.e2e[run.profiles[0]]["latency_ms"]["model_init_ms"]
    sweep = read_json(run.dir / "llama_ngl_sweep.json")
    sweep_source = run
    if sweep is None and baseline is not None:
        sweep, sweep_source = read_json(baseline.dir / "llama_ngl_sweep.json"), baseline
    sweep_part: list[str] = []
    if sweep:
        sweep_rows = [
            [
                f"{r['ngl']}" + (" (выбрано)" if r["ngl"] == sweep["selected_ngl"] else ""),
                ms(r["llama_only_vram_mib"]),
                ms(r["llama_vram_after_mib"]),
                " → ".join(f"{q['prompt_tok_s']:.0f}" for q in r["requests"]),
                " → ".join(f"{q['gen_tok_s']:.1f}" for q in r["requests"]),
                "да" if r["startup_fit_warning"] else "нет",
            ]
            for r in sweep["runs"]
        ]
        write_csv(
            "llama_ngl_sweep.csv",
            ["ngl", "request", "prompt_tokens", "prompt_tok_s", "gen_tokens", "gen_tok_s"],
            [
                [
                    r["ngl"],
                    i + 1,
                    q["prompt_tokens"],
                    q["prompt_tok_s"],
                    q["gen_tokens"],
                    q["gen_tok_s"],
                ]
                for r in sweep["runs"]
                for i, q in enumerate(r["requests"])
            ],
        )
        sweep_part = [
            "### Выбор числа GPU-слоёв LLM",
            "",
            *(
                [
                    f"Замер выполнен перед базовым прогоном `{sweep_source.rel}` на том же стенде.",
                    "",
                ]
                if sweep_source is not run
                else []
            ),
            f"Три последовательных запроса конвейера (реранкинг → генерация → NLI) при процессе "
            f"PyTorch с моделями на той же карте ({ms(sweep['pytorch_process_reserved_mib'])} МиБ "
            f"reserved); llama.cpp сборки {sweep['llama_cpp_build']}, флаги "
            "`-c 4096 -b 128 -ub 64 -np 1 --cache-ram 0`. Скорости из журнала llama-server, ток/с по "
            "запросам:",
            "",
            table(
                [
                    "-ngl",
                    "VRAM llama, МиБ",
                    "После прогона, МиБ",
                    "Промпт, ток/с",
                    "Генерация, ток/с",
                    "Предупреждение «failed to fit»",
                ],
                sweep_rows,
                "lrrllr",
            ),
            "",
            "Уменьшение VRAM llama-server после прогона означает, что драйвер вытеснил её буферы в "
            "общую память: со второго запроса скорость падает. Сквозной прогон выполнен с "
            f"`-ngl {sweep['selected_ngl']}`.",
            "",
        ]
    return "\n".join(
        [
            "## 6. Задержки и ресурсы",
            "",
            *sweep_part,
            "",
            "Запросы, дошедшие до генерации (без первого запроса с ленивой загрузкой моделей); мс:",
            "",
            table(["Профиль", "Этап", "n", "Среднее", "p50", "p95"], rows, "llrrrr"),
            "",
            *ttft_note,
            f"![Время этапов]({fig})",
            "",
            f"Однократная загрузка моделей при первом запросе: {', '.join(ms(x) for x in init) or 'н/д'} мс. "
            "Отказ по порогу релевантности обходится без вызова LLM и NLI.",
            "",
            "Память (пики за прогон, МиБ; GPU = вся карта по nvidia-smi, включая llama-server):",
            "",
            table(
                [
                    "Прогон",
                    "GPU, вся карта",
                    "PyTorch max allocated",
                    "RSS процесса оценки",
                    "RSS llama-server",
                ],
                res_rows,
            ),
            "",
        ]
    )


def cyrillic_share(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    return sum(bool(re.match(r"[А-Яа-яЁё]", c)) for c in letters) / len(letters) if letters else 0.0


def section_limits(run: Run) -> str:
    language: list[str] = []
    profile = (
        "guard_on" if "guard_on" in run.profiles else (run.profiles[0] if run.profiles else None)
    )
    if profile:
        drafts = [r for r in run.e2e_cases[profile] if r.get("ok") and r.get("raw_answer")]
        ru = sum(cyrillic_share(r["raw_answer"]) > 0.5 for r in drafts)
        en_q = [r for r in drafts if r["language"] == "en"]
        en_q_ru = sum(cyrillic_share(r["raw_answer"]) > 0.5 for r in en_q)
        language = [
            f"- Все документы корпуса англоязычные. При `RAG_GENERATION_LANGUAGE=ru` {ru} из "
            f"{len(drafts)} черновиков написаны преимущественно кириллицей (на английские вопросы: "
            f"{en_q_ru} из {len(en_q)}), поэтому NLI в основном сравнивает русскую гипотезу с "
            "английской посылкой; на контрольных парах доля ложной отбраковки для EN- и "
            "RU-гипотез приведена в разделе 4.2.",
        ]
    return "\n".join(
        [
            "## 7. Ограничения",
            "",
            "- *Vanilla RAG* в этом отчёте - черновик той же LLM с тем же промптом (включая "
            "premise guard в прод-профиле) и тем же порогом релевантности, а не внешняя базовая "
            "система; сравнение изолирует вклад NLI-фильтра.",
            *language,
            "- Разметка датасета подготовлена ИИ; механические проверки (дословные цитаты, "
            "отсутствие терминов в корпусе, дедупликация) не заменяют экспертную проверку смысла.",
            "- Корректность ответа оценивается по ключевым фактам (`answer_must_contain`), а не "
            "экспертом: ответ с верным числом, но лишним неподтверждённым утверждением считается "
            "верным; правильный ответ другими словами может быть засчитан как неверный.",
            "- Галлюцинации на уровне кейса определены для вопросов без ответа и adversarial; "
            "лишние неподтверждённые утверждения в ответах на отвечаемые вопросы этой метрикой не "
            "учитываются (их видно в статусах утверждений).",
            "- Один прогон на профиль при temperature 0 и фиксированном seed; разброс между "
            "прогонами не измерялся.",
            "- Корпус открытый (RFC, NIST, PostgreSQL, PEP), вопросы RU/EN; ГОСТ/СП и другие "
            "русскоязычные нормативные документы не измерялись.",
            *(
                [
                    "- Пороги верификатора подобраны на dev-половине (110-111 кейсов); test-половина "
                    "того же размера даёт оценку переноса, но не заменяет независимый датасет.",
                    "- NLI проверяет выводимость утверждения из источника, а не то, отвечает ли "
                    "оно на вопрос: верный по источнику смежный факт проходит фильтр. Утверждение, "
                    "добавляющее субъект, которого нет в окне («A QUIC client ...», «RFC 9110 "
                    "specifies ...»), mDeBERTa часто оценивает как Neutral, хотя чанк взят из "
                    "этого документа; служебный заголовок чанка содержит хэш документа, а не "
                    "название, поэтому дать модели этот контекст без переиндексации нельзя.",
                    "- `replay` повторяет только первую попытку генерации и требует локальных "
                    "логов `full/` исходного прогона (не хранятся в git).",
                ]
                if run.setting("nli_window_scoring")
                else []
            ),
            "",
        ]
    )


def section_repro(run: Run, baseline: Run | None = None) -> str:
    source = (run.ablation or {}).get("source_run") or (baseline.rel if baseline else None)
    return "\n".join(
        [
            "## 8. Воспроизведение",
            "",
            "```powershell",
            "python scripts/build_eval_dataset_v3.py            # пересобрать и проверить датасет",
            ".\\tools\\llama\\llama-server.exe -m .\\models\\llm\\Qwen2.5-14B-Instruct-Q4_K_M.gguf -ngl 25 -c 4096 -b 128 -ub 64 -np 1 --cache-ram 0 --port 8080 --host 127.0.0.1",
            f"python evaluate_suite.py --run-dir {run.rel} retrieval",
            f"python evaluate_suite.py --run-dir {run.rel} nli",
            f"python evaluate_suite.py --run-dir {run.rel} e2e --profile guard_on",
            f"python evaluate_suite.py --run-dir {run.rel} e2e --profile guard_off",
            *(
                [f"python evaluate_suite.py --run-dir {run.rel} replay --source {source}"]
                if source
                else []
            ),
            f"python scripts/render_report.py {run.rel}"
            + (f" --baseline {baseline.rel}" if baseline else ""),
            "```",
            "",
            "`replay` не вызывает LLM и поиск: он заново проверяет сохранённые черновики "
            "(`full/e2e_<profile>.jsonl`, не хранятся в git) и NLI-пробы и поэтому выполняется за "
            "минуты; им удобно проверять любое изменение верификатора до полного прогона.",
            "",
            "Режимы выполняются последовательно: встроенный Qdrant допускает одного владельца "
            "индекса. Индекс `.artifacts/eval-v3-index` строится при первом запуске и проверяется "
            "при каждом следующем (число чанков и наличие всех эталонных `chunk_id`).",
            "",
        ]
    )


def headline(run: Run, baseline: Run | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"run": run.rel, "commit": run.config.get("git", {}).get("commit")}
    if run.retrieval:
        out["retrieval_recall@5"] = {m: run.retrieval["overall"][m]["recall@5"] for m in MODE_ORDER}
        out["retrieval_mrr@10"] = {m: run.retrieval["overall"][m]["mrr@10"] for m in MODE_ORDER}
        out["retrieval_recall@5_prod_by_slice"] = {
            **{c: v["hybrid_rerank"]["recall@5"] for c, v in run.retrieval["by_category"].items()},
            **{
                f"lang_{lang}": v["hybrid_rerank"]["recall@5"]
                for lang, v in run.retrieval["by_language"].items()
            },
        }
    if run.nli:
        d = run.nli["overall"]["decision_rule"]
        out["nli"] = {
            "pairs": run.nli["overall"]["pairs"],
            "catch_rate": d["hallucination_catch_rate"],
            "false_rejection_rate": d["false_rejection_rate"],
            "false_rejection_rate_by_language": {
                lang: b["decision_rule"]["false_rejection_rate"]
                for lang, b in run.nli["by_hypothesis_language"].items()
            },
            "by_split": {
                split: {
                    "catch_rate": m["hallucination_catch_rate"],
                    "false_rejection_rate": m["false_rejection_rate"],
                }
                for split, m in run.nli.get("by_split", {}).items()
            },
            "macro_f1": run.nli["overall"]["argmax"]["macro_f1"],
            "latency_ms_p50": run.nli["latency_ms_per_pair"]["p50"],
        }
    for profile in run.profiles:
        s = run.e2e[profile]
        out[f"e2e_{profile}"] = {
            variant: {
                k: s[variant][k]["rate"]
                for k in (
                    "hallucination",
                    "false_refusal_on_answerable",
                    "key_fact_accuracy_on_answerable",
                    "correct",
                )
            }
            | {
                "correct_k": s[variant]["correct"]["k"],
                "hallucination_k": s[variant]["hallucination"]["k"],
                "hallucination_n": s[variant]["hallucination"]["n"],
            }
            for variant in ("vanilla", "factrag")
        } | {
            "filter_induced_hallucinations": s["hallucination_ids"]["factrag_only"],
            "by_split": {
                split: {variant: block[variant]["correct"]["rate"] for variant in block}
                for split, block in s.get("by_split", {}).items()
            },
            "hallucination_relative_reduction": s["hallucination_relative_reduction"],
            "mcnemar_p": s["paired_correctness"]["mcnemar_exact_p"],
            "paired_correctness": s["paired_correctness"],
            "claims_by_status_reason": s["claims"].get("by_status_reason"),
            "latency_p50_ms": {
                k: v["p50"] for k, v in s["latency_ms"]["generated_queries"].items()
            },
            "latency_p95_ms": {
                k: v["p95"] for k, v in s["latency_ms"]["generated_queries"].items()
            },
        }
    if run.ablation:
        out["verifier_replay"] = {
            "source_run": run.ablation["source_run"],
            "steps": [
                {
                    "step": step["step"],
                    **{
                        profile: {
                            "factrag_correct": e2e["factrag"]["correct"]["k"],
                            "false_refusals_on_answerable": e2e["factrag"][
                                "false_refusal_on_answerable"
                            ]["k"],
                            "factrag_hallucinations": e2e["factrag"]["hallucination"]["k"],
                            "filter_induced_hallucinations": len(
                                e2e["filter_induced_hallucinations"]
                            ),
                        }
                        for profile, e2e in step["e2e"].items()
                    },
                    "nli_false_rejection_rate": step["nli"]["overall"]["false_rejection_rate"],
                    "nli_catch_rate": step["nli"]["overall"]["hallucination_catch_rate"],
                }
                for step in run.ablation["steps"]
            ],
        }
    if baseline:
        out["baseline"] = headline(baseline)
    return out


def readme_block(run: Run, baseline: Run | None = None) -> str:
    """Compact results table injected into README.md between the results markers."""
    lines = []
    prod = run.e2e.get("guard_on")
    base = baseline.e2e.get("guard_on") if baseline else None
    if prod:
        v, f = prod["vanilla"], prod["factrag"]
        lat = prod["latency_ms"]["generated_queries"]
        metrics = (
            ("Корректный исход, все вопросы ↑", "correct"),
            ("Ключевые факты верны (отвечаемые) ↑", "key_fact_accuracy_on_answerable"),
            ("Отказы на отвечаемых вопросах ↓", "false_refusal_on_answerable"),
            ("Галлюцинации на вопросах с известной истиной ↓", "hallucination"),
        )
        rows = [
            [title, rate_cell(v[key]), rate_cell(f[key])]
            + ([rate_cell(base["factrag"][key])] if base else [])
            for title, key in metrics
        ]
        induced = len(prod["hallucination_ids"]["factrag_only"])
        rows.append(
            ["Галлюцинации, внесённые NLI-фильтром ↓", "-", str(induced)]
            + ([str(len(base["hallucination_ids"]["factrag_only"]))] if base else [])
        )
        pc = prod["paired_correctness"]
        header = ["Метрика", "Standard RAG (без NLI)", "FactRAG-NLI"] + (
            [f"FactRAG-NLI, базовый прогон `{baseline.rel.rsplit('/', 1)[-1]}`"]
            if base and baseline
            else []
        )
        lines += [
            f"Сквозной прогон на {prod['cases']} вопросах датасета v3, прод-профиль. *Standard RAG* "
            "- черновик той же LLM до NLI-фильтра (тот же поиск, порог релевантности и промпт), "
            "поэтому сравнение парное.",
            "",
            table(header, rows, "l" + "r" * (len(header) - 1)),
            "",
            "NLI-фильтр "
            + direction(
                v["correct"]["rate"],
                f["correct"]["rate"],
                ("снижает", "повышает", "не меняет"),
            )
            + " долю корректных исходов относительно черновика (только Standard RAG верен в "
            f"{plural(pc['vanilla_only_correct'], 'кейсе', 'кейсах', 'кейсах')}, только "
            f"FactRAG-NLI в {pc['factrag_only_correct']}; точный тест Мак-Немара p = "
            f"{pval(pc['mcnemar_exact_p'])}) и "
            + (
                f"уменьшает число галлюцинаций с {v['hallucination']['k']} до "
                f"{f['hallucination']['k']} из {f['hallucination']['n']}"
                if f["hallucination"]["k"] < v["hallucination"]["k"]
                else f"не уменьшает число галлюцинаций ({v['hallucination']['k']} → "
                f"{f['hallucination']['k']} из {f['hallucination']['n']})"
            )
            + ". Что изменено по сравнению с базовым прогоном и вклад каждого изменения: "
            "[раздел 2 отчёта](reports/EVALUATION_REPORT.md#2-что-изменено-и-почему); разбор "
            "оставшихся ошибок: [раздел 5](reports/EVALUATION_REPORT.md#5-разбор-ошибок). "
            f"NLI-верификация добавляет p50 {ms(lat['nli_ms']['p50'])} мс к запросу; основное время "
            f"занимает генерация LLM (p50 {ms(lat['generation_ms']['p50'])} мс).",
            "",
        ]
    if run.nli:
        d = run.nli["overall"]["decision_rule"]
        lines.append(
            f"NLI-модуль на {run.nli['overall']['pairs']} контрольных парах EN/RU: перехват "
            f"{pct(d['hallucination_catch_rate'])}, ложная отбраковка {pct(d['false_rejection_rate'])}"
            + (
                f" (базовый прогон: {pct(baseline.nli['overall']['decision_rule']['hallucination_catch_rate'])} "
                f"и {pct(baseline.nli['overall']['decision_rule']['false_rejection_rate'])})"
                if baseline and baseline.nli
                else ""
            )
            + "."
        )
        lines.append("")
    if run.retrieval:
        o = run.retrieval["overall"]
        lines.append(
            f"Поиск, Recall@5: dense {num(o['dense']['recall@5'])}, BM25 {num(o['sparse']['recall@5'])}, "
            f"RRF {num(o['hybrid_rrf']['recall@5'])}, RRF + реранкинг {num(o['hybrid_rerank']['recall@5'])}"
            + (
                f" (базовый прогон: {num(baseline.retrieval['overall']['hybrid_rerank']['recall@5'])})"
                if baseline and baseline.retrieval
                else ""
            )
            + "."
        )
    return "\n".join(lines)


def update_readme(block: str) -> bool:
    path = ROOT / "README.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    if start not in text or end not in text:
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{block}\n{end}{tail}", encoding="utf-8", newline="\n")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run_dir", type=Path)
    cli.add_argument(
        "--baseline", type=Path, default=None, help="Earlier run directory to compare against"
    )
    args = cli.parse_args(argv)
    run = Run(args.run_dir.resolve())
    baseline = Run(args.baseline.resolve()) if args.baseline else None
    # Every table and figure is regenerated; files of a previous render must not linger.
    for stale in [*TABLES.glob("*.csv"), *FIGURES.glob("*.svg")]:
        stale.unlink()
    dataset = load_dataset(ROOT / "data" / "eval_dataset_v3.json")
    lengths = premise_lengths(dataset)
    head = headline(run, baseline)
    parts = [
        "# Оценка FactRAG-NLI на датасете v3",
        "",
        f"Прогон [`{run.rel}`]({run.dir.relative_to(REPORTS).as_posix()}), дата "
        f"{run.config.get('created_utc', 'н/д')[:10]}, "
        f"коммит `{(run.config.get('git', {}).get('commit') or 'н/д')[:12]}`. Все числа "
        "сгенерированы `scripts/render_report.py` из сырых логов прогона.",
        "",
        section_summary(run, dataset, head, baseline),
        section_changes(run, baseline),
        section_method(run, dataset),
        "## 4. Количественные результаты",
        "",
        section_retrieval(run),
        section_nli(run),
        section_e2e(run),
        section_errors(run, dataset, lengths),
        section_latency(run, baseline),
        section_limits(run),
        section_repro(run, baseline),
    ]
    REPORTS.mkdir(exist_ok=True)
    text = "\n".join(parts).replace("\n\n\n", "\n\n")
    (REPORTS / "EVALUATION_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    (REPORTS / "summary.json").write_text(
        json.dumps(head, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    readme = update_readme(readme_block(run, baseline))
    print(
        f"Wrote reports/EVALUATION_REPORT.md, reports/summary.json, {len(list(TABLES.glob('*.csv')))} tables, {len(list(FIGURES.glob('*.svg')))} figures"
        + ("; README results block updated" if readme else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
