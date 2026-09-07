"""Inspectable local RAG demo; exported evidence is from the same query execution."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from evaluate_quality import GoldenDataset
from generator_verifier import ClaimStatus
from pipeline import RAGPipeline, RAGResponse
from settings import RAGSettings

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "golden": ROOT / "data/golden_dataset.json",
    "holdout": ROOT / "data/holdout_dataset.json",
}
_CONTROL = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f]")


def safe_text(value: str) -> str:
    """Do not execute terminal controls or Rich markup found in retrieved documents."""
    return _CONTROL.sub("", value)


@dataclass(frozen=True)
class Preset:
    id: str
    question: str


def load_presets(dataset: str) -> list[Preset]:
    data = GoldenDataset.model_validate_json(DATASETS[dataset].read_text(encoding="utf-8"))
    return [Preset(e.id, e.question) for e in data.examples]


def select_question(value: str, presets: Sequence[Preset]) -> str:
    for preset in presets:
        if preset.id == value:
            return preset.question
    if value.isdecimal() and 1 <= int(value) <= len(presets):
        return presets[int(value) - 1].question
    raise ValueError(f"Неизвестный пресет: {value}")


def show_presets(console: Console, presets: Sequence[Preset]) -> None:
    table = Table(title="Пресеты: в генератор передаётся только вопрос")
    table.add_column("№")
    table.add_column("ID")
    table.add_column("Вопрос")
    for i, preset in enumerate(presets, 1):
        table.add_row(str(i), preset.id, safe_text(preset.question))
    console.print(table)


def ensure_index(pipeline: RAGPipeline) -> dict[str, Any]:
    """Opening the retriever validates Qdrant and restores its BM25 cache first."""
    count = pipeline.retriever.size
    if count:
        return {"action": "reused", "chunks": count}
    root = pipeline.settings.document_root
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
    )
    if not paths:
        raise ValueError(f"Нет документов для индексации: {root}")
    stats = pipeline.index_documents(paths)
    if not pipeline.retriever.size:
        raise ValueError("Документы не дали ни одного индексируемого чанка")
    return {"action": "indexed", **stats.model_dump(mode="json")}


def retrieval_snapshot(pipeline: RAGPipeline, response: RAGResponse) -> list[dict[str, Any]]:
    details = {d["chunk_id"]: d for d in response.retrieval_diagnostics}
    included = {s.chunk_id: s.source_id for s in response.sources}
    result = []
    candidate_ids = response.reranked_candidate_ids or response.hybrid_candidate_ids
    for key in candidate_ids[:5]:
        chunk = pipeline.retriever.get_chunk(key)
        if chunk is None:
            raise RuntimeError(f"Чанк из трассы запроса отсутствует в индексе: {key}")
        result.append(
            {
                "stage": "reranked" if response.reranked_candidate_ids else "hybrid_before_filter",
                "chunk_id": key,
                "source": chunk.source,
                "section_path": list(chunk.section_path),
                "text": chunk.text,
                "source_id": included.get(key),
                **details.get(key, {}),
            }
        )
    return result


def render(console: Console, response: RAGResponse, retrieved: list[dict[str, Any]]) -> None:
    console.rule("1. Поиск и реранкинг — результат этого запроса")
    console.print(Text(safe_text(response.query), style="bold"))
    if not retrieved:
        console.print("Нет кандидатов после фильтрации релевантности.")
    for i, hit in enumerate(retrieved, 1):
        scores = f"{hit.get('stage', 'reranked')}: RRF={float(hit.get('rrf_score', 0)):.6f}; rerank={float(hit.get('rerank_score', 0)):.4f}"
        label = hit.get("source_id") or "не включён в контекст LLM"
        body = f"{hit['source']}\n{' > '.join(hit['section_path'])}\n{scores}; {label}\n\n{hit['text']}"
        console.print(Panel(Text(safe_text(body)), title=f"Top-{i} · {hit['chunk_id']}"))
    console.rule("2. Генерация и итоговый ответ")
    if response.raw_answer and response.raw_answer != response.answer:
        console.print(
            Panel(
                Text(safe_text(response.raw_answer)),
                title="Черновик LLM — НЕ подтверждённый ответ",
                border_style="yellow",
            )
        )
    console.print(
        Panel(
            Text(safe_text(response.answer)),
            title="ОТКАЗ" if response.refused else "Ответ после NLI",
            border_style="red" if response.refused else "green",
        )
    )
    if response.refusal_reason:
        console.print(Text(f"Причина: {response.refusal_reason}"))
    console.rule("3. Проверка утверждений NLI")
    for claim in response.claims:
        label, color = {
            ClaimStatus.VERIFIED: ("ENTAILED", "green"),
            ClaimStatus.CONTRADICTION: ("CONTRADICTED", "red"),
            ClaimStatus.UNVERIFIED: ("UNVERIFIED", "yellow"),
        }[claim.status]
        console.print(Text(f"[{label}] {safe_text(claim.claim)}", style=color))
        for evidence in claim.evidence:
            p = evidence.probabilities
            if p is None:
                console.print(Text(f"  {evidence.source_id}: NLI не выполнена — {evidence.reason}"))
            else:
                # Raw classifier label differs from the policy decision on thresholds/citations.
                raw_label = max(
                    (p.p_entailment, "ENTAILED"),
                    (p.p_neutral, "NEUTRAL"),
                    (p.p_contradiction, "CONTRADICTED"),
                )[1]
                console.print(
                    Text(
                        f"  {evidence.source_id}: {raw_label}; E={p.p_entailment:.3f} N={p.p_neutral:.3f} C={p.p_contradiction:.3f}"
                    )
                )
        if claim.reason:
            console.print(Text(f"  Решение фильтра: {claim.reason}"))
    if not response.claims:
        console.print("Утверждений для NLI нет. Это не означает подтверждение ответа.")
    console.rule("4. Время стадий")
    table = Table("Стадия", "мс")
    for key, title in [
        ("model_init_ms", "Загрузка моделей"),
        ("search_ms", "Hybrid retrieval"),
        ("rerank_ms", "Cross-Encoder"),
        ("generation_ms", "Генерация"),
        ("llm_ttft_ms", "TTFT"),
        ("nli_ms", "NLI"),
        ("total_ms", "Всего"),
    ]:
        table.add_row(
            title,
            f"{response.execution_stats[key]:.1f}"
            if key in response.execution_stats
            else "не измерено",
        )
    console.print(table)
    console.print(
        "NLI — автоматический фильтр, не экспертная гарантия истинности. Стадии показаны после выполнения."
    )


def export_demo(
    path: Path, response: RAGResponse, retrieved: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Export an offline HTML view plus exact machine-readable evidence."""
    path = path.resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Для --export укажите путь с расширением .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "kind": "rag_demo_not_quality_evaluation",
        "retrieved_top5": retrieved,
        "response": response.model_dump(mode="json"),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    import io

    view = Console(file=io.StringIO(), record=True, width=110, markup=False, highlight=False)
    render(view, response, retrieved)
    html = path.with_suffix(".html")
    view.save_html(str(html), clear=False)
    return path, html


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    choice = cli.add_mutually_exclusive_group()
    choice.add_argument("--query")
    choice.add_argument("--preset")
    cli.add_argument("--dataset", choices=list(DATASETS), default="golden")
    cli.add_argument("--list-presets", action="store_true")
    cli.add_argument("--ensure-index", action="store_true")
    cli.add_argument("--index-only", action="store_true")
    cli.add_argument("--export", type=Path)
    cli.add_argument("--no-color", action="store_true")
    args = cli.parse_args(argv)
    if args.export and args.export.suffix.lower() != ".json":
        cli.error("--export должен оканчиваться на .json")
    if args.index_only and (args.query is not None or args.preset or args.export):
        cli.error("--index-only несовместим с запросом/экспортом")
    console = Console(no_color=args.no_color, markup=False, highlight=False)
    try:
        presets = (
            load_presets(args.dataset)
            if args.list_presets or args.preset or (args.query is None and not args.index_only)
            else []
        )
        if args.list_presets:
            show_presets(console, presets)
            return 0
        question = select_question(args.preset, presets) if args.preset else args.query
        # Validate preset before loading models or modifying the index.
        with RAGPipeline(RAGSettings()) as pipeline:
            if args.ensure_index or args.index_only:
                console.print(
                    Text(json.dumps(ensure_index(pipeline), ensure_ascii=False)), soft_wrap=True
                )
            if args.index_only:
                return 0
            if question is not None:
                response = pipeline.query(question)
                retrieved = retrieval_snapshot(pipeline, response)
                render(console, response, retrieved)
                if args.export:
                    export_demo(args.export, response, retrieved)
                return 0
            show_presets(console, presets)
            console.print(
                "Введите вопрос, номер пресета, /preset ID, /list или /quit. Экспорт перезаписывается последним успешным запросом."
            )
            while True:
                try:
                    value = console.input("Вопрос > ").strip()
                    if value in {"/quit", "/exit"}:
                        break
                    if value == "/list":
                        show_presets(console, presets)
                        continue
                    if not value:
                        continue
                    question = (
                        select_question(value[8:].strip(), presets)
                        if value.startswith("/preset ")
                        else select_question(value, presets)
                        if value.isdecimal()
                        else value
                    )
                    response = pipeline.query(question)
                    retrieved = retrieval_snapshot(pipeline, response)
                    render(console, response, retrieved)
                    if args.export:
                        export_demo(args.export, response, retrieved)
                except (EOFError, KeyboardInterrupt):
                    break
                except (ValueError, RuntimeError, OSError) as exc:
                    console.print(Text(f"Ошибка: {safe_text(str(exc))}", style="red"))
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Ошибка: {safe_text(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
