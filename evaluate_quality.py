"""Local labeled-corpus evaluation with explicit metric denominators and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import sys
import importlib.metadata
from pathlib import Path
from typing import Any, Literal, Sequence, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from generator_verifier import ClaimStatus
from pipeline import RAGPipeline, RAGResponse
from settings import RAGSettings


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CorpusDocument(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_id: str
    url: str | None = None


class Judgment(StrictModel):
    document_id: str
    chunk_id: str | None = None
    grade: int = Field(default=1, ge=1, le=3)


class QAExample(StrictModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_url: str
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ground_truth_answer: str
    relevant_fragment_quote: str
    created_by: Literal["ai_pregenerated", "human"] = "ai_pregenerated"
    kind: Literal["fact", "synthesis", "negative"] = "fact"
    relevant: list[Judgment]
    is_negative: bool = False
    review_status: Literal["ai_draft", "human_reviewed"] = "ai_draft"

    @property
    def reference_answer(self) -> str:
        return self.ground_truth_answer

    @property
    def unanswerable(self) -> bool:
        return self.is_negative

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not self.question.strip():
            raise ValueError("Blank question")
        if self.unanswerable:
            if self.relevant or self.reference_answer.strip():
                raise ValueError("Negative examples require empty reference and relevance")
        elif not self.relevant or not self.reference_answer.strip():
            raise ValueError("Positive examples require reference answer and relevance")
        keys = [(r.document_id, r.chunk_id) for r in self.relevant]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate relevance judgment")
        return self


class GoldenDataset(StrictModel):
    version: Literal[1] = 1
    description: str
    corpus_root: str
    documents: list[CorpusDocument] = Field(min_length=1)
    examples: list[QAExample] = Field(min_length=1)
    parser_settings: dict[
        Literal["chunk_target_tokens", "chunk_max_tokens", "chunk_overlap_tokens"], int
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len({e.id for e in self.examples}) != len(self.examples):
            raise ValueError("Duplicate example ID")
        if len({d.document_id for d in self.documents}) != len(self.documents):
            raise ValueError("Duplicate document identity")
        doc_ids = {d.document_id for d in self.documents}
        if any(j.document_id not in doc_ids for e in self.examples for j in e.relevant):
            raise ValueError("Unknown relevant document")
        for example in self.examples:
            if not any(
                d.sha256 == example.document_sha256 and d.url == example.source_url
                for d in self.documents
            ):
                raise ValueError("Example source URL/hash does not match manifest")
        return self


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ranking_metrics(ranked: Sequence[str], grades: dict[str, int], k: int) -> dict[str, float]:
    """Chunk-level metrics; a document judgment marks all of its chunks relevant."""
    if k < 1 or not grades:
        raise ValueError("Positive k and nonempty relevance required")
    ranks = list(dict.fromkeys(ranked))[:k]
    relevant_ranks = [i for i, key in enumerate(ranks, 1) if key in grades]
    dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(i + 1) for i, key in enumerate(ranks, 1))
    ideal = sum(
        (2**grade - 1) / math.log2(i + 1)
        for i, grade in enumerate(sorted(grades.values(), reverse=True)[:k], 1)
    )
    return {
        "hit_at_k": float(bool(relevant_ranks)),
        "mrr_at_k": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        "ndcg_at_k": dcg / ideal,
    }


def cosine(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Embedding dimensions differ or are empty")
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norm = math.sqrt(sum(a * a for a in first) * sum(b * b for b in second))
    if not norm or not math.isfinite(norm) or not math.isfinite(dot):
        raise ValueError("Invalid embedding norm")
    return max(-1.0, min(1.0, dot / norm))


def citation_metrics(response: RAGResponse, grades: dict[str, int]) -> dict[str, float | None]:
    """Final-answer citation occurrences: relevant chunk AND NLI-supported claim."""
    citations = re.findall(r"\[(S[1-9]\d*)\]", response.answer)
    sources = {s.source_id: s.chunk_id for s in response.sources}
    supported = {
        sid
        for claim in response.claims
        if claim.status == ClaimStatus.VERIFIED
        for sid in claim.source_ids
    }
    correct = [sources[sid] for sid in citations if sid in supported and sources.get(sid) in grades]
    return {
        "citation_precision": len(correct) / len(citations) if citations else None,
        "citation_recall": len(set(correct)) / len(grades) if grades else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for row in rows for name in row.get("metrics", {})})
    summary: dict[str, Any] = {
        "cases": len(rows),
        "errors": sum(not row["ok"] for row in rows),
        "metrics": {},
    }
    for name in names:
        values = [
            row["metrics"][name] for row in rows if row.get("metrics", {}).get(name) is not None
        ]
        summary["metrics"][name] = {
            "mean": sum(values) / len(values) if values else None,
            "defined_cases": len(values),
            "undefined_cases": len(rows) - len(values),
        }
    claims = [claim for row in rows for claim in row.get("response", {}).get("claims", [])]
    verified = sum(claim["status"] == "verified" for claim in claims)
    summary["faithfulness_micro"] = {
        "verified_claims": verified,
        "total_claims": len(claims),
        "percent": 100 * verified / len(claims) if claims else None,
    }
    attempted_claims = [
        claim
        for row in rows
        for attempt in (
            row.get("response", {}).get("verification_attempts") or [row.get("response", {})]
        )
        for claim in attempt.get("claims", [])
    ]
    summary["faithfulness_all_attempts"] = {
        "verified_claims": sum(c["status"] == "verified" for c in attempted_claims),
        "total_claims": len(attempted_claims),
    }
    for negative, name in ((True, "negative_refusal"), (False, "positive_answer")):
        cases = [row for row in rows if row.get("unanswerable") is negative]
        success = sum(
            row["ok"] and row.get("response", {}).get("refused") == negative for row in cases
        )
        summary[name] = {
            "successful": success,
            "cases": len(cases),
            "rate": success / len(cases) if cases else None,
        }
    return summary


def evaluate(pipeline: RAGPipeline, dataset: GoldenDataset, k: int) -> dict[str, Any]:
    if not 1 <= k <= pipeline.settings.rerank_candidates:
        raise ValueError("k outside rerank candidate limit")
    paths = []
    for doc in dataset.documents:
        path = pipeline.settings.resolve_document(doc.path)
        if digest(path) != doc.sha256:
            raise ValueError(f"Corpus hash mismatch: {doc.path}")
        paths.append(path)
    pipeline.index_documents(paths)
    chunks = {c.chunk_id: c for d in pipeline.documents.values() for c in d.chunks}
    if {d.document_id for d in dataset.documents} != set(pipeline.documents):
        raise ValueError("Document IDs changed; rebuild dataset using current parser settings")
    labels: dict[str, dict[str, int]] = {}
    for example in dataset.examples:
        grades: dict[str, int] = {}
        for judgment in example.relevant:
            matches = [
                c.chunk_id
                for c in chunks.values()
                if c.document_id == judgment.document_id
                and (judgment.chunk_id is None or c.chunk_id == judgment.chunk_id)
            ]
            if not matches:
                raise ValueError(f"Unresolved relevance in {example.id}")
            for key in matches:
                grades[key] = max(grades.get(key, 0), judgment.grade)
        labels[example.id] = grades
    rows: list[dict[str, Any]] = []
    for example in dataset.examples:
        print(
            f"Evaluating {example.id} ({len(rows) + 1}/{len(dataset.examples)})",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        metrics: dict[str, float | None] = {}
        row: dict[str, Any] = {
            "id": example.id,
            "question": example.question,
            "reference_answer": example.reference_answer,
            "review_status": example.review_status,
            "unanswerable": example.unanswerable,
            "ok": False,
            "metrics": metrics,
        }
        grades = labels[example.id]
        try:
            response = pipeline.query(example.question, top_k=k)
            row["response"] = response.model_dump(mode="json")
            row["refusal_reason"] = response.refusal_reason
            for stage, ranked in (
                ("hybrid", response.hybrid_candidate_ids),
                ("reranked", response.reranked_candidate_ids),
            ):
                if grades:
                    metrics.update(
                        {
                            f"{stage}_{name}": value
                            for name, value in ranking_metrics(ranked, grades, k).items()
                        }
                    )
            metrics["faithfulness"] = response.faithfulness_score if response.claims else None
            metrics["refusal_correct"] = float(response.refused == example.unanswerable)
            metrics.update(citation_metrics(response, grades))
            metrics["answer_relevance_cosine"] = None
            metrics["reference_answer_cosine"] = None
            if not response.refused:
                answer = re.sub(r"\[S\d+\]", "", response.answer)
                texts = [answer, example.question]
                if example.reference_answer:
                    texts.append(example.reference_answer)
                vectors = pipeline.retriever.encoder.encode(texts)
                metrics["answer_relevance_cosine"] = cosine(vectors[0], vectors[1])
                if len(vectors) == 3:
                    metrics["reference_answer_cosine"] = cosine(vectors[0], vectors[2])
            row["ok"] = True
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            metrics.clear()
            metrics["refusal_correct"] = 0.0
            if grades:
                for stage in ("hybrid", "reranked"):
                    metrics.update(
                        {f"{stage}_{name}": 0.0 for name in ("hit_at_k", "mrr_at_k", "ndcg_at_k")}
                    )
        row["wall_ms"] = (time.perf_counter() - started) * 1000
        rows.append(row)
    return {
        "version": 1,
        "k": k,
        "dataset_status": "human_reviewed"
        if all(e.review_status == "human_reviewed" for e in dataset.examples)
        else "contains_ai_drafts_not_expert_validated",
        "fallback_allowed": pipeline.settings.allow_fallback,
        "dataset": dataset.model_dump(mode="json"),
        "definitions": {
            "retrieval": "Chunk-level graded labels; hybrid before rerank, reranked after threshold; negatives excluded",
            "faithfulness": "NLI verified / all final-generation claims; no claims undefined; prior attempts retained and counted separately",
            "answer_relevance": "BGE cosine(answer, question), not calibrated factual accuracy",
            "citations": "Final answer occurrences; relevant AND NLI supported; recall over relevant chunks",
            "errors": "Retrieval/refusal scored zero; unavailable semantic metrics null with explicit counts",
        },
        "summary": summarize(rows),
        "rows": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {report['dataset_status']}",
        "",
        "| Metric | Mean | Defined | Undefined |",
        "|---|---:|---:|---:|",
    ]
    for name, value in report["summary"]["metrics"].items():
        mean = "n/a" if value["mean"] is None else f"{value['mean']:.4f}"
        lines.append(f"| {name} | {mean} | {value['defined_cases']} | {value['undefined_cases']} |")
    lines.append(f"\nErrors: {report['summary']['errors']} / {report['summary']['cases']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", default=Path("evaluation_report.json"), type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        path = args.dataset.resolve(strict=True)
        dataset_bytes = path.read_bytes()
        dataset = GoldenDataset.model_validate_json(dataset_bytes)
        root = (path.parent / dataset.corpus_root).resolve(strict=True)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        index = output.parent / f"evaluation-index-{time.time_ns()}"
        settings = RAGSettings.model_validate(
            {
                **RAGSettings().model_dump(),
                **dataset.parser_settings,
                "document_root": root,
                "qdrant_path": index,
                "embedding_batch_size": 2,
                "rerank_batch_size": 2,
            }
        )
        if settings.allow_fallback:
            raise ValueError("Quality CLI requires real models: allow_fallback=False")
        with RAGPipeline(settings) as pipeline:
            report = evaluate(pipeline, dataset, args.k)
        report["dataset_sha256"] = hashlib.sha256(dataset_bytes).hexdigest()
        report["runtime_versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "qdrant-client",
                "pydantic",
            )
        }
        report["configuration"] = {
            name: str(getattr(settings, name))
            for name in (
                "model_device",
                "model_precision",
                "llm_backend",
                "llm_context_window",
                "max_tokens",
                "rerank_score_threshold",
                "rerank_rrf_alpha",
                "rerank_window_tokens",
                "rrf_k",
                "dense_weight",
                "sparse_weight",
                "llm_tokenizer_mode",
                "generation_language",
                "generation_repair_attempts",
                "nli_evidence_focus",
                "nli_entailment_threshold",
                "nli_contradiction_threshold",
                "embedding_model_path",
                "reranker_model_path",
                "nli_model_path",
                "llm_model_path",
            )
        }
        report["index_path"] = str(index)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(markdown(report))
        return 1 if report["summary"]["errors"] else 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
