"""Command-line entrypoint for the enhanced RAG pipeline, not a dense-only baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from generator_verifier import ClaimStatus
from pipeline import RAGPipeline, RAGResponse

REFERENCE_QUESTIONS = (
    "Какое допустимое давление указано в документах?",
    "Какие температурные ограничения указаны в документах?",
    "Какие исключения предусмотрены требованиями?",
)


def display(response: RAGResponse, *, color: bool) -> None:
    """Render only the cleaned answer; filtered claims remain clearly marked."""
    print(response.answer)
    for claim in response.claims:
        verified = claim.status == ClaimStatus.VERIFIED
        label = "ПОДТВЕРЖДЕНО" if verified else "ОТФИЛЬТРОВАНО"
        line = f"[{label}] {claim.claim}"
        print(("\033[32m" if verified else "\033[31m") + line + "\033[0m" if color else line)
    for source in response.sources:
        print(f"[{source.source_id}] {source.source}: {source.citation}; score={source.score:.5f}")
    print(
        f"Подтверждено утверждений: {response.faithfulness_score:.3f}; {response.execution_stats['total_ms']:.1f} мс"
    )


def benchmark(pipeline: RAGPipeline, questions: Sequence[str]) -> dict[str, Any]:
    """Record all wall times, including failed/refused queries; no quality estimates."""
    if not questions:
        raise ValueError("Benchmark question list is empty")
    records: list[dict[str, Any]] = []
    timings: list[float] = []
    for question in questions:
        started = time.perf_counter()
        try:
            response = pipeline.query(question)
            record: dict[str, Any] = {
                "question": question,
                "ok": True,
                "response": response.model_dump(mode="json"),
            }
        except (ValueError, RuntimeError, OSError) as exc:
            record = {"question": question, "ok": False, "error": str(exc)}
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        records.append({**record, "wall_ms": elapsed})
    ordered = sorted(timings)
    return {
        "kind": "latency_smoke_not_ragas",
        "count": len(records),
        "failed": sum(not record["ok"] for record in records),
        "p50_ms": ordered[math.ceil(len(ordered) * 0.5) - 1],
        "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "includes_cold_start_and_errors": True,
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local hybrid RAG CLI; benchmark measures latency only"
    )
    parser.add_argument("--index", type=Path, help="Document directory within RAG_DOCUMENT_ROOT")
    parser.add_argument("--query")
    parser.add_argument(
        "--summarize", metavar="DOC_ID", help="Independent document summarization route"
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-file", type=Path, help="JSON array of reference question strings"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON lines")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    if not (
        args.index or args.query is not None or args.benchmark or args.interactive or args.summarize
    ):
        parser.error("Specify --index, --query, --summarize, --interactive or --benchmark")
    if args.benchmark_file and not args.benchmark:
        parser.error("--benchmark-file requires --benchmark")
    if args.summarize and (
        args.index or args.query is not None or args.interactive or args.benchmark
    ):
        parser.error("--summarize cannot be combined with QA/index/benchmark")
    try:
        if args.summarize:
            from settings import RAGSettings
            from summarization import SummarizationPipeline

            with SummarizationPipeline(RAGSettings()) as summarizer:
                summary = summarizer.summarize(args.summarize)
            print(summary.model_dump_json() if args.json else summary.summary)
            if not args.json:
                print("Саммари выборки: neutral допускается; это не подтверждение всех фактов.")
            return 0
        with RAGPipeline.from_environment() as pipeline:
            if args.index:
                directory = args.index.resolve(strict=True)
                if not directory.is_dir() or not directory.is_relative_to(
                    pipeline.settings.document_root
                ):
                    raise ValueError("Index directory must be inside RAG_DOCUMENT_ROOT")
                paths = sorted(
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
                )
                if not paths:
                    raise ValueError("Index directory contains no supported documents")
                print(pipeline.index_documents(paths).model_dump_json())
            if args.query is not None:
                response = pipeline.query(args.query)
                if args.json:
                    print(response.model_dump_json())
                else:
                    display(response, color=sys.stdout.isatty() and not args.no_color)
            if args.interactive:
                while True:
                    try:
                        question = input("Вопрос (/exit для выхода): ").strip()
                        if question == "/exit":
                            break
                        if not question:
                            continue
                        response = pipeline.query(question)
                        if args.json:
                            print(response.model_dump_json())
                        else:
                            display(response, color=sys.stdout.isatty() and not args.no_color)
                    except (EOFError, KeyboardInterrupt):
                        break
                    except (ValueError, RuntimeError, OSError) as exc:
                        print(f"Ошибка: {exc}", file=sys.stderr)
            if args.benchmark:
                questions = list(REFERENCE_QUESTIONS)
                if args.benchmark_file:
                    data = json.loads(args.benchmark_file.read_text(encoding="utf-8"))
                    if (
                        not isinstance(data, list)
                        or not data
                        or any(not isinstance(q, str) or not q.strip() for q in data)
                    ):
                        raise ValueError(
                            "Benchmark file must contain a nonempty array of questions"
                        )
                    questions = data
                result = benchmark(pipeline, questions)
                print(json.dumps(result, ensure_ascii=False, allow_nan=False))
                if result["failed"]:
                    return 1
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
