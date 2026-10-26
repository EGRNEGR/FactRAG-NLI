"""Offline benchmark harness with optional Ragas integration.

The fallback metrics are deterministic and dependency-free; when Ragas is installed,
``run_ragas`` can be used with an explicitly supplied local LLM/embeddings adapter.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from pipeline import RAGPipeline


@dataclass(slots=True, frozen=True)
class BenchmarkCase:
    question: str
    ground_truth: str
    source_chunk_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    faithfulness: float
    context_recall: float
    context_precision: float
    answer_relevancy: float


def synthesize_cases(pipeline: RAGPipeline, *, max_cases: int = 50) -> list[BenchmarkCase]:
    """Create transparent extractive QA cases from indexed chunks."""
    cases: list[BenchmarkCase] = []
    for document in pipeline.documents.values():
        for chunk in document.chunks:
            sentences = _sentences(chunk.text)
            if not sentences:
                continue
            fact = sentences[0]
            section = " > ".join(chunk.section_path) or "документ"
            cases.append(
                BenchmarkCase(f"Что указано в разделе «{section}»?", fact, (chunk.chunk_id,))
            )
            if len(cases) >= max_cases:
                return cases
    return cases


def evaluate_fallback(pipeline: RAGPipeline, cases: Sequence[BenchmarkCase]) -> BenchmarkResult:
    """Calculate useful token-overlap proxies without external model calls."""
    if not cases:
        raise ValueError("benchmark has no cases")
    faithfulness: list[float] = []
    recall: list[float] = []
    precision: list[float] = []
    relevancy: list[float] = []
    for case in cases:
        response = pipeline.query(case.question)
        answer_terms = _terms(response.answer)
        truth_terms = _terms(case.ground_truth)
        source_terms = (
            set().union(*(_terms(source.text) for source in response.sources))
            if response.sources
            else set()
        )
        faithfulness.append(_coverage(answer_terms, source_terms))
        recall.append(_coverage(truth_terms, source_terms))
        precision.append(_coverage(source_terms, truth_terms))
        relevancy.append(_coverage(_terms(case.question), answer_terms))
    return BenchmarkResult(
        *(sum(values) / len(values) for values in (faithfulness, recall, precision, relevancy))
    )


def run_ragas(cases: Sequence[BenchmarkCase], pipeline: RAGPipeline) -> Any:
    """Run Ragas when installed; the caller must supply local Ragas-compatible adapters."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except ImportError as exc:
        raise RuntimeError("Install the evaluation extra to run Ragas") from exc
    rows = []
    for case in cases:
        response = pipeline.query(case.question)
        rows.append(
            {
                "question": case.question,
                "answer": response.answer,
                "contexts": [source.text for source in response.sources],
                "ground_truth": case.ground_truth,
            }
        )
    dataset = Dataset.from_list(rows)
    # Modern Ragas accepts metric objects; model/embeddings are intentionally not configured here.
    return evaluate(
        dataset, metrics=[Faithfulness(), ContextRecall(), ContextPrecision(), AnswerRelevancy()]
    )


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), re.UNICODE))


def _coverage(expected: set[str], actual: set[str]) -> float:
    return len(expected & actual) / max(len(expected), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local RAG with fallback or Ragas metrics"
    )
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--max-cases", type=int, default=50)
    args = parser.parse_args()
    pipeline = RAGPipeline.from_environment()
    for path in args.documents:
        pipeline.ingest_path(path)
    cases = synthesize_cases(pipeline, max_cases=args.max_cases)
    result = run_ragas(cases, pipeline) if args.ragas else evaluate_fallback(pipeline, cases)
    print(
        json.dumps(
            asdict(result) if isinstance(result, BenchmarkResult) else dict(result),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
