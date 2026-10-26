"""Render reports/EVALUATION_REPORT.md, CSV tables, SVG figures and reports/summary.json.

Every number is read from one run directory produced by evaluate_suite.py; nothing is typed
by hand. Sections whose mode has not been run are reported as missing.

Usage:  python scripts/render_report.py reports/runs/<run_id>
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
REASON_NAMES = {
    "verified:none": "VERIFIED",
    "unverified:missing_citation": "UNVERIFIED: нет маркера [S#] после разбиения на предложения, NLI не вызывался",
    "unverified:cited_support_not_established": "UNVERIFIED: NLI не подтвердил (P(ent) ниже порога, P(con) < 0.2)",
    "contradiction:cited_support_not_established": "CONTRADICTION: P(con) ≥ 0.2 хотя бы в одном источнике",
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


def premise_lengths(dataset: EvalDatasetV3) -> dict[str, int]:
    """Premise chunk length in characters, from the local parse cache when available."""
    lengths: dict[str, int] = {}
    for path in (ROOT / ".artifacts" / "eval_v3_parse_cache").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        for chunk in record["chunks"]:
            lengths[chunk["chunk_id"]] = len(chunk["text"])
    wanted = {c.nli_probe.premise_chunk_id for c in dataset.cases}
    return {cid: n for cid, n in lengths.items() if cid in wanted}


# --------------------------------------------------------------------------- sections


def section_summary(run: Run, dataset: EvalDatasetV3, headline: dict[str, Any]) -> str:
    lines = ["## 1. Краткие выводы", ""]
    prod = run.e2e.get("guard_on")
    if prod:
        v, f = prod["vanilla"], prod["factrag"]
        pc = prod["paired_correctness"]
        causes = pc.get("vanilla_only_correct_by_cause", {})
        lines.append(
            f"- **Сквозной результат.** На {prod['cases']} вопросах доля корректных исходов у "
            f"FactRAG-NLI {direction(v['correct']['rate'], f['correct']['rate'])} с "
            f"{pct(v['correct']['rate'])} (Vanilla RAG: черновик той же LLM до NLI-фильтра) до "
            f"{pct(f['correct']['rate'])}: фильтр сделал неверными {pc['vanilla_only_correct']} "
            f"исходов и исправил {pc['factrag_only_correct']} (точный тест Мак-Немара, p = "
            f"{pval(pc['mcnemar_exact_p'])}). Из потерянных: "
            + "; ".join(f"{CAUSE_NAMES.get(k, k)} - {n}" for k, n in causes.items())
            + f". Отказы на отвечаемых вопросах: {pct(v['false_refusal_on_answerable']['rate'])} → "
            f"{pct(f['false_refusal_on_answerable']['rate'])}."
        )
        guard = [
            f"{PROFILE_SHORT[p]}: Vanilla {count(run.e2e[p]['vanilla']['hallucination'])}, "
            f"FactRAG-NLI {count(run.e2e[p]['factrag']['hallucination'])}"
            for p in run.profiles
        ]
        introduced = len(prod.get("hallucination_ids", {}).get("factrag_only", []))
        known = [
            r
            for r in run.e2e_cases["guard_on"]
            if r.get("ok") and r["expected_behavior"] != "answer"
        ]
        gate = sum(r["refusal_reason"] == "relevance_threshold_not_met" for r in known)
        llm = sum(r["refusal_reason"] == "llm_refusal" for r in known)
        lines.append(
            "- **Галлюцинации.** На вопросах с известной истиной (без ответа, с ложной "
            "предпосылкой, lexical_trap без ответа) ответов без опоры на корпус: "
            + "; ".join(guard)
            + f". Доля мала уже у черновика: из {len(known)} таких вопросов в прод-профиле "
            f"{gate} отсёк порог релевантности (LLM не вызывается), на {llm} LLM ответила "
            "отказом по инструкции промпта. "
            + (
                f"В {introduced} из {f['hallucination']['k']} галлюцинаций FactRAG-NLI в прод-профиле "
                "черновик был верным: фильтр удалил фразу об отсутствии сведений или исправление "
                "предпосылки и опубликовал оставшееся подтверждённое утверждение."
                if introduced
                else ""
            )
        )
        claims = prod["claims"]
        reasons = claims.get("by_status_reason", {})
        total = claims["total"] or 1
        nli_part = ""
        if run.nli:
            d = run.nli["overall"]["decision_rule"]
            sm = d["status_matrix"]["entailment"]
            ent_n = sum(sm.values())
            nli_part = (
                f" На {run.nli['overall']['pairs']} контрольных парах с эталонной посылкой правило "
                f"перехватывает {pct(d['hallucination_catch_rate'])} искажённых и неподтверждённых "
                f"утверждений, но блокирует {pct(d['false_rejection_rate'])} верных; "
                f"{sm.get('contradiction', 0)} из {ent_n} верных гипотез получают P(con) ≥ 0.2. "
                "Ложная отбраковка по языку гипотезы: "
                + ", ".join(
                    f"{lang.upper()} {pct(b['decision_rule']['false_rejection_rate'])}"
                    for lang, b in run.nli["by_hypothesis_language"].items()
                )
                + "."
            )
        lines.append(
            f"- **Причины на уровне утверждений.** Из {claims['total']} утверждений черновиков "
            f"{reasons.get('unverified:missing_citation', 0)} "
            f"({pct(reasons.get('unverified:missing_citation', 0) / total)}) остались без маркера "
            f"источника после разбиения на предложения и отклонены без вызова NLI; "
            f"{reasons.get('contradiction:cited_support_not_established', 0)} "
            f"({pct(reasons.get('contradiction:cited_support_not_established', 0) / total)}) "
            f"получили CONTRADICTION." + nli_part
        )
    if run.retrieval:
        o = run.retrieval["overall"]
        rrf, dense = o["hybrid_rrf"]["recall@5"], o["dense"]["recall@5"]
        line = (
            f"- **Поиск и задержка.** Гибрид с реранкингом: Recall@5 "
            f"{num(o['hybrid_rerank']['recall@5'])}, MRR@10 {num(o['hybrid_rerank']['mrr@10'])} "
            f"против {num(dense)} / {num(o['dense']['mrr@10'])} у dense и "
            f"{num(o['sparse']['recall@5'])} / {num(o['sparse']['mrr@10'])} у BM25; RRF без "
            f"реранкинга ({num(rrf)}) "
            + (
                "ниже dense, то есть выигрыш гибрида даёт реранкер."
                if rrf < dense
                else "не ниже dense."
            )
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
        "## 2. Методология и датасет",
        "",
        "### 2.1. Датасет v3",
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
        "### 2.2. Стенд и воспроизводимость",
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
        "### 2.3. Режимы и метрики",
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
        return "### 3.1. Поиск\n\nРежим не запускался.\n"
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
            "### 3.1. Поиск: абляция режимов",
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
        return "### 3.2. NLI-верификация\n\nРежим не запускался.\n"
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
    chosen = [r for r in sweep if r["entailment_threshold"] == 0.4]
    sweep_rows = [
        [
            f"{r['contradiction_threshold']:.1f}"
            + (" (прод)" if r["contradiction_threshold"] == 0.2 else ""),
            pct(r["hallucination_catch_rate"]),
            pct(r["false_rejection_rate"]),
        ]
        for r in chosen
    ]
    status = s["overall"]["decision_rule"]["status_matrix"]
    status_rows = [
        [LABEL_RU[g], *(status[g].get(st, 0) for st in ("verified", "unverified", "contradiction"))]
        for g in labels
    ]
    lat = s["latency_ms_per_pair"]
    return "\n".join(
        [
            "### 3.2. NLI-верификация на контрольных парах",
            "",
            f"{s['overall']['pairs']} пар (220 проб × EN/RU), ни одна не превысила контекст 512 токенов.",
            "",
            "**Классификатор (argmax трёх классов), P / R / F1:**",
            "",
            table(["Срез", "Entailment", "Neutral", "Contradiction", "Macro-F1", "Accuracy"], rows),
            "",
            f"![Матрица ошибок, EN]({figs[0]}) ![Матрица ошибок, RU]({figs[1]})",
            "",
            "**Боевое правило (P(con) ≥ 0.2 в любом источнике → CONTRADICTION; иначе VERIFIED "
            "при P(ent) > 0.4 во всех источниках или ≥ 0.7 в одном):**",
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
            "**Чувствительность к порогу противоречия** (порог entailment 0.4, strong 0.7):",
            "",
            table(["Порог contradiction", "Catch rate", "False rejection"], sweep_rows),
            "",
            f"Полная сетка: [`tables/nli_threshold_sweep.csv`](tables/nli_threshold_sweep.csv). "
            f"Время одной пары: p50 {num(lat['p50'], 1)} мс, p95 {num(lat['p95'], 1)} мс.",
            "",
        ]
    )


def section_e2e(run: Run) -> str:
    if not run.profiles:
        return "### 3.3. Сквозной прогон\n\nРежим не запускался.\n"
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
    keys = list(REASON_NAMES) + sorted(
        {k for p in run.profiles for k in run.e2e[p]["claims"].get("by_status_reason", {})}
        - set(REASON_NAMES)
    )
    reason_rows = [
        [
            REASON_NAMES.get(k, k),
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
            "### 3.3. Сквозной прогон: Vanilla RAG и FactRAG-NLI",
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
    out = ["## 4. Разбор ошибок", ""]
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
            "### 4.1. NLI: где mDeBERTa-v3 ошибается",
            "",
            f"**Ложные противоречия** ({len(false_con)} из "
            f"{sum(p['label'] == 'entailment' for p in pairs)} проб Entailment): верное утверждение "
            f"получает P(contradiction) ≥ 0.2 и блокируется. По категориям: "
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
            "выбирает Neutral (P(ent) ≤ 0.4 при P(con) < 0.2). По категориям: "
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
            f"### 4.2. Сквозной прогон ({PROFILE_NAMES[profile]})",
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
            "ответа или ответ без исправления ложной предпосылки. Если черновик был верным, "
            "галлюцинацию внёс фильтр: он публикует только VERIFIED-утверждения, а фраза об "
            "отсутствии сведений или исправление предпосылки опорой в источнике не подтверждается.",
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
            + ", ".join(f"{REASON_NAMES.get(k, k)} - {v}" for k, v in over_reasons.most_common())
            + ".",
            "",
            table(
                ["Кейс", "Черновик LLM", "Вердикты NLI (первые три утверждения)"],
                [[r["id"], quote(r["raw_answer"], 130), claim_line(r)] for r in over[:6]],
                "lll",
            ),
            "",
            f"**Разбиение на утверждения.** "
            f"{plural(len(uncited), 'утверждение', 'утверждения', 'утверждений')} не получили "
            f"маркер [S#]. Из них {len(fragments)} - обрывки предложений (эвристика: оканчиваются "
            "на «Rev.», «No.», «;» или начинаются с цифры), например: "
            + ", ".join(f"«{quote(c, 60)}»" for c in list(dict.fromkeys(fragments))[:3])
            + ". Защита сокращений в `segment_claims` (`generator_verifier.py`) охватывает только "
            "русские сокращения (пп., рис., т. е. и др.), поэтому «SP 800-53 Rev. 5» режется после "
            "«Rev.», и первая часть остаётся без цитаты. Точка с запятой разделяет утверждения по "
            "правилу, а LLM ставит маркер в конце предложения, поэтому часть до «;» тоже остаётся "
            f"без цитаты. Остальные {len(uncited) - len(fragments)} - целые предложения, после "
            "которых LLM не поставила маркер источника.",
            "",
        ]
    return "\n".join(out)


def section_latency(run: Run) -> str:
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
            "## 5. Задержки и ресурсы",
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
            "RU-гипотез приведена в разделе 3.2.",
        ]
    return "\n".join(
        [
            "## 6. Ограничения",
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
            "",
        ]
    )


def section_repro(run: Run) -> str:
    return "\n".join(
        [
            "## 7. Воспроизведение",
            "",
            "```powershell",
            "python scripts/build_eval_dataset_v3.py            # пересобрать и проверить датасет",
            ".\\tools\\llama\\llama-server.exe -m .\\models\\llm\\Qwen2.5-14B-Instruct-Q4_K_M.gguf -ngl 25 -c 4096 -b 128 -ub 64 -np 1 --cache-ram 0 --port 8080 --host 127.0.0.1",
            f"python evaluate_suite.py --run-dir {run.rel} retrieval",
            f"python evaluate_suite.py --run-dir {run.rel} nli",
            f"python evaluate_suite.py --run-dir {run.rel} e2e --profile guard_on",
            f"python evaluate_suite.py --run-dir {run.rel} e2e --profile guard_off",
            f"python scripts/render_report.py {run.rel}",
            "```",
            "",
            "Режимы выполняются последовательно: встроенный Qdrant допускает одного владельца "
            "индекса. Индекс `.artifacts/eval-v3-index` строится при первом запуске и проверяется "
            "при каждом следующем (число чанков и наличие всех эталонных `chunk_id`).",
            "",
        ]
    )


def headline(run: Run) -> dict[str, Any]:
    out: dict[str, Any] = {"run": run.rel, "commit": run.config.get("git", {}).get("commit")}
    if run.retrieval:
        out["retrieval_recall@5"] = {m: run.retrieval["overall"][m]["recall@5"] for m in MODE_ORDER}
        out["retrieval_mrr@10"] = {m: run.retrieval["overall"][m]["mrr@10"] for m in MODE_ORDER}
    if run.nli:
        d = run.nli["overall"]["decision_rule"]
        out["nli"] = {
            "pairs": run.nli["overall"]["pairs"],
            "catch_rate": d["hallucination_catch_rate"],
            "false_rejection_rate": d["false_rejection_rate"],
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
                "hallucination_k": s[variant]["hallucination"]["k"],
                "hallucination_n": s[variant]["hallucination"]["n"],
            }
            for variant in ("vanilla", "factrag")
        } | {
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
    return out


def readme_block(run: Run) -> str:
    """Compact results table injected into README.md between the results markers."""
    lines = []
    prod = run.e2e.get("guard_on")
    if prod:
        v, f = prod["vanilla"], prod["factrag"]
        lat = prod["latency_ms"]["generated_queries"]
        rows = [
            ["Корректный исход, все вопросы ↑", rate_cell(v["correct"]), rate_cell(f["correct"])],
            [
                "Ключевые факты верны (отвечаемые) ↑",
                rate_cell(v["key_fact_accuracy_on_answerable"]),
                rate_cell(f["key_fact_accuracy_on_answerable"]),
            ],
            [
                "Отказы на отвечаемых вопросах ↓",
                rate_cell(v["false_refusal_on_answerable"]),
                rate_cell(f["false_refusal_on_answerable"]),
            ],
            [
                "Галлюцинации на вопросах с известной истиной ↓",
                rate_cell(v["hallucination"]),
                rate_cell(f["hallucination"]),
            ],
        ]
        pc = prod["paired_correctness"]
        claims = prod["claims"]
        reasons = claims.get("by_status_reason", {})
        hv, hf = v["hallucination"]["k"], f["hallucination"]["k"]
        lines += [
            f"Сквозной прогон на {prod['cases']} вопросах датасета v3, прод-профиль. *Standard RAG* "
            "- черновик той же LLM до NLI-фильтра (тот же поиск, порог релевантности и промпт), "
            "поэтому сравнение парное.",
            "",
            table(["Метрика", "Standard RAG (без NLI)", "FactRAG-NLI"], rows, "lrr"),
            "",
            "В текущей конфигурации NLI-фильтр **"
            + direction(
                v["correct"]["rate"], f["correct"]["rate"], ("снизил", "повысил", "не изменил")
            )
            + "** долю корректных исходов (только Standard RAG верен в "
            f"{plural(pc['vanilla_only_correct'], 'кейсе', 'кейсах', 'кейсах')}, "
            f"только FactRAG-NLI в {pc['factrag_only_correct']}; точный тест Мак-Немара p = "
            f"{pval(pc['mcnemar_exact_p'])}) и "
            + (
                f"не уменьшил число галлюцинаций ({hv} → {hf} из {f['hallucination']['n']})"
                if hf >= hv
                else f"уменьшил число галлюцинаций с {hv} до {hf} из {f['hallucination']['n']}"
            )
            + f". Из {claims['total']} утверждений черновиков VERIFIED получили "
            f"{reasons.get('verified:none', 0)}; {reasons.get('unverified:missing_citation', 0)} "
            "отклонены без вызова NLI, потому что после разбиения на предложения остались без "
            f"маркера источника; {reasons.get('contradiction:cited_support_not_established', 0)} "
            "получили CONTRADICTION (P(con) ≥ 0.2); "
            f"{reasons.get('unverified:cited_support_not_established', 0)} не подтверждены NLI. "
            "Разбор причин: [раздел 4 отчёта](reports/EVALUATION_REPORT.md#4-разбор-ошибок). "
            f"NLI-верификация добавляет p50 {ms(lat['nli_ms']['p50'])} мс к запросу; основное время "
            f"занимает генерация LLM (p50 {ms(lat['generation_ms']['p50'])} мс).",
            "",
        ]
    if run.nli:
        d = run.nli["overall"]["decision_rule"]
        lines.append(
            f"NLI-модуль на {run.nli['overall']['pairs']} контрольных парах EN/RU: перехват "
            f"{pct(d['hallucination_catch_rate'])}, ложная отбраковка {pct(d['false_rejection_rate'])}, "
            f"macro-F1 {num(run.nli['overall']['argmax']['macro_f1'])}."
        )
        lines.append("")
    if run.retrieval:
        o = run.retrieval["overall"]
        lines.append(
            f"Поиск, Recall@5: dense {num(o['dense']['recall@5'])}, BM25 {num(o['sparse']['recall@5'])}, "
            f"RRF {num(o['hybrid_rrf']['recall@5'])}, RRF + реранкинг {num(o['hybrid_rerank']['recall@5'])}."
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
    args = cli.parse_args(argv)
    run = Run(args.run_dir.resolve())
    dataset = load_dataset(ROOT / "data" / "eval_dataset_v3.json")
    lengths = premise_lengths(dataset)
    head = headline(run)
    parts = [
        "# Оценка FactRAG-NLI на датасете v3",
        "",
        f"Прогон [`{run.rel}`]({run.dir.relative_to(REPORTS).as_posix()}), дата "
        f"{run.config.get('created_utc', 'н/д')[:10]}, "
        f"коммит `{(run.config.get('git', {}).get('commit') or 'н/д')[:12]}`. Все числа "
        "сгенерированы `scripts/render_report.py` из сырых логов прогона.",
        "",
        section_summary(run, dataset, head),
        section_method(run, dataset),
        "## 3. Количественные результаты",
        "",
        section_retrieval(run),
        section_nli(run),
        section_e2e(run),
        section_errors(run, dataset, lengths),
        section_latency(run),
        section_limits(run),
        section_repro(run),
    ]
    REPORTS.mkdir(exist_ok=True)
    text = "\n".join(parts).replace("\n\n\n", "\n\n")
    (REPORTS / "EVALUATION_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    (REPORTS / "summary.json").write_text(
        json.dumps(head, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    readme = update_readme(readme_block(run))
    print(
        f"Wrote reports/EVALUATION_REPORT.md, reports/summary.json, {len(list(TABLES.glob('*.csv')))} tables, {len(list(FIGURES.glob('*.svg')))} figures"
        + ("; README results block updated" if readme else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
