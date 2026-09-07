"""Lazy, resource-owning local RAG orchestration with typed public responses."""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from document_parser import DocumentParser, ParsedDocument
from generator_verifier import (
    ClaimStatus,
    GeneratorVerifier,
    GenerationMetadata,
    LocalLLMGenerator,
    NLIFactVerifier,
    REFUSAL,
    VerifiedClaim,
)
from hybrid_retriever import HybridRetriever
from settings import RAGSettings


class PipelineError(RuntimeError):
    """A component failed; the exception is not converted to a factual refusal."""


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class IndexingStats(_Result):
    processed_files: int = Field(ge=0)
    indexed_chunks: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    document_ids: list[str]


class IndexingError(PipelineError):
    """A batch failed after the completed documents in stats were committed."""

    def __init__(self, path: Path, stats: IndexingStats) -> None:
        super().__init__(
            f"Indexing failed for {path}; {stats.processed_files} files already committed"
        )
        self.stats = stats


class SourceCitation(_Result):
    source_id: str
    chunk_id: str
    document_id: str
    source: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    citation: str
    text: str
    score: float
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float
    rerank_score: float | None
    final_score: float | None = None
    metadata: dict[str, Any]

    @field_validator("metadata")
    @classmethod
    def json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize nested tuples to their persisted JSON representation."""
        result: dict[str, Any] = json.loads(json.dumps(value, allow_nan=False))
        return result


ClaimVerification = VerifiedClaim


class RAGResponse(_Result):
    query: str
    answer: str
    sources: list[SourceCitation]
    faithfulness_score: float = Field(ge=0, le=1)
    claims: list[ClaimVerification]
    execution_stats: dict[str, float]
    refused: bool
    is_reliable: bool
    hybrid_candidate_ids: list[str] = Field(default_factory=list)
    reranked_candidate_ids: list[str] = Field(default_factory=list)
    generation_metadata: GenerationMetadata | None = None
    raw_answer: str | None = None
    refusal_reason: str | None = None
    retrieval_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    verification_attempts: list[dict[str, Any]] = Field(default_factory=list)


QueryResponse = RAGResponse

_INJECTION = re.compile(
    r"(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|prompts)|"
    r"(?:игнорируй|забудь|отмени)\s+(?:все\s+)?(?:предыдущие|системные|прежние)\s+инструкции|"
    r"<\|(?:im_start|im_end|system|assistant)\|>|\[INST\]|<<SYS>>",
    re.I,
)


class RAGPipeline:
    """Serialize workflows. Injected components transfer their lifecycle to this object.

    Initialization is lazy, including LLM/NLI: empty retrieval never loads them.
    Batch indexing commits per document, not transactionally across a whole batch.
    """

    def __init__(
        self,
        settings: RAGSettings | None = None,
        *,
        parser: DocumentParser | None = None,
        retriever: HybridRetriever | None = None,
        generator: LocalLLMGenerator | None = None,
        fact_verifier: NLIFactVerifier | None = None,
        verifier: GeneratorVerifier | None = None,
    ) -> None:
        try:
            configured = settings or RAGSettings()
            self.settings = RAGSettings.model_validate(configured.model_dump())
        except Exception as exc:
            raise PipelineError(f"Invalid pipeline settings: {exc}") from exc
        if verifier is not None and (generator is not None or fact_verifier is not None):
            raise ValueError("Specify either the legacy verifier facade or separate components")
        self._parser, self._retriever = parser, retriever
        self._generator, self._fact_verifier = generator, fact_verifier
        self._legacy_verifier = verifier
        self._lock = threading.RLock()
        self._closed = False
        self.documents: dict[str, ParsedDocument] = {}

    @classmethod
    def from_environment(cls) -> RAGPipeline:
        return cls(RAGSettings())

    def _ready(self) -> None:
        if self._closed:
            raise PipelineError("Pipeline is closed")

    @property
    def parser(self) -> DocumentParser:
        with self._lock:
            self._ready()
            if self._parser is None:
                self._parser = DocumentParser.from_settings(self.settings)
            return self._parser

    @property
    def retriever(self) -> HybridRetriever:
        with self._lock:
            self._ready()
            if self._retriever is None:
                self._retriever = HybridRetriever(settings=self.settings)
            return self._retriever

    def ingest_path(self, path: str | Path, *, backend: str = "auto") -> ParsedDocument:
        with self._lock:
            self._ready()
            resolved = self.settings.resolve_document(path)
            parsed = self.parser.parse(resolved, backend=backend)
            self.retriever.add(parsed.chunks)
            self.documents[parsed.document_id] = parsed
            return parsed

    def index_documents(self, paths: list[Path]) -> IndexingStats:
        """Validate all paths first; report committed work if later parsing fails."""
        with self._lock:
            self._ready()
            started = time.perf_counter()
            resolved = list(dict.fromkeys(self.settings.resolve_document(path) for path in paths))
            ids: list[str] = []
            chunks = 0
            for path in resolved:
                try:
                    parsed = self.ingest_path(path)
                except Exception as exc:
                    raise IndexingError(
                        path,
                        IndexingStats(
                            processed_files=len(ids),
                            indexed_chunks=chunks,
                            elapsed_ms=(time.perf_counter() - started) * 1000,
                            document_ids=ids,
                        ),
                    ) from exc
                ids.append(parsed.document_id)
                chunks += len(parsed.chunks)
            return IndexingStats(
                processed_files=len(ids),
                indexed_chunks=chunks,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                document_ids=ids,
            )

    def ingest_text(
        self, text: str, *, source: str = "memory", document_id: str | None = None
    ) -> ParsedDocument:
        with self._lock:
            self._ready()
            parsed = self.parser.parse_text(text, source=source, document_id=document_id)
            self.retriever.add(parsed.chunks)
            self.documents[parsed.document_id] = parsed
            return parsed

    def _question(self, question: str) -> str:
        normalized = unicodedata.normalize("NFKC", question).strip()
        if not normalized or len(normalized) > self.settings.max_question_chars:
            raise ValueError("Question is empty or exceeds max_question_chars")
        if any(unicodedata.category(c) in {"Cc", "Cf"} and c not in "\n\t\r" for c in normalized):
            raise ValueError("Question contains forbidden control characters")
        if _INJECTION.search(normalized):
            raise ValueError("Question contains an instruction-override pattern")
        return normalized

    def query(self, question: str, *, top_k: int | None = None) -> RAGResponse:
        """Retrieve, generate and verify; scores describe raw claims before filtering."""
        question = self._question(question)
        with self._lock:
            self._ready()
            started = time.perf_counter()
            stats = {
                "search_ms": 0.0,
                "rerank_ms": 0.0,
                "generation_ms": 0.0,
                "nli_ms": 0.0,
                "model_init_ms": 0.0,
                "total_ms": 0.0,
            }
            init = time.perf_counter()
            retriever = self.retriever
            stats["model_init_ms"] += (time.perf_counter() - init) * 1000
            results, timing, hybrid_ids, diagnostics = retriever.search_trace(question, top_k=top_k)
            stats.update(timing)
            results = [
                hit
                for hit in results
                if hit.rerank_score is not None
                and hit.rerank_score >= self.settings.rerank_score_threshold
            ]
            if not results:
                stats["total_ms"] = (time.perf_counter() - started) * 1000
                return RAGResponse(
                    query=question,
                    answer=REFUSAL,
                    sources=[],
                    claims=[],
                    faithfulness_score=0,
                    execution_stats=stats,
                    refused=True,
                    is_reliable=False,
                    hybrid_candidate_ids=hybrid_ids,
                    refusal_reason="relevance_threshold_not_met",
                    retrieval_diagnostics=diagnostics,
                )
            attempts: list[dict[str, Any]] = []
            if self._legacy_verifier is not None:
                verified = self._legacy_verifier.answer(question, results)
            else:
                init = time.perf_counter()
                if self._generator is None:
                    self._generator = LocalLLMGenerator(self.settings)
                if self._fact_verifier is None:
                    try:
                        self._fact_verifier = NLIFactVerifier(
                            self.settings, evidence_encoder=retriever.encoder
                        )
                    except Exception:
                        self._generator.close()
                        self._generator = None
                        raise
                stats["model_init_ms"] += (time.perf_counter() - init) * 1000
                raw = self._generator.generate(question, results)
                verified = self._fact_verifier.verify(raw)
                attempts.append(verified.model_dump(mode="json"))
                for _ in range(self.settings.generation_repair_attempts):
                    if not verified.refused or not verified.claims:
                        break
                    raw = self._generator.generate(
                        question,
                        results,
                        revision={
                            "previous_answer": verified.raw_text,
                            "rejected_claims": [
                                {"text": claim.claim, "status": claim.status.value}
                                for claim in verified.claims
                                if claim.status != ClaimStatus.VERIFIED
                            ],
                        },
                    )
                    verified = self._fact_verifier.verify(raw)
                    attempts.append(verified.model_dump(mode="json"))
            stats.update(
                generation_ms=sum(a["generation_metadata"]["generation_ms"] for a in attempts)
                if attempts
                else verified.generation_ms,
                nli_ms=sum(a["verification_ms"] for a in attempts)
                if attempts
                else verified.verification_ms,
            )
            if verified.generation_metadata.ttft_ms is not None:
                stats["llm_ttft_ms"] = verified.generation_metadata.ttft_ms
            sources: list[SourceCitation] = []
            for index, hit in enumerate(results, 1):
                source_id = f"S{index}"
                context = verified.contexts.get(source_id)
                if context is None:
                    continue
                chunk = hit.chunk
                if context.chunk_id != chunk.chunk_id or context.text != chunk.text:
                    raise PipelineError("Generator context map does not match retrieved evidence")
                sources.append(
                    SourceCitation(
                        source_id=source_id,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        source=chunk.source,
                        section_path=chunk.section_path,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        citation=chunk.citation,
                        text=context.text,
                        score=hit.score,
                        rrf_score=hit.rrf_score,
                        dense_score=hit.dense_score,
                        sparse_score=hit.sparse_score,
                        rerank_score=hit.rerank_score,
                        final_score=hit.final_score,
                        metadata=dict(chunk.metadata),
                    )
                )
            supported = sum(claim.status == ClaimStatus.VERIFIED for claim in verified.claims)
            stats["total_ms"] = (time.perf_counter() - started) * 1000
            return RAGResponse(
                query=question,
                answer=verified.cleaned_text,
                sources=sources,
                claims=list(verified.claims),
                faithfulness_score=supported / len(verified.claims) if verified.claims else 0,
                execution_stats=stats,
                refused=verified.refused,
                is_reliable=verified.is_reliable,
                hybrid_candidate_ids=hybrid_ids,
                reranked_candidate_ids=[hit.chunk.chunk_id for hit in results],
                generation_metadata=verified.generation_metadata,
                raw_answer=verified.raw_text,
                retrieval_diagnostics=diagnostics,
                verification_attempts=attempts,
                refusal_reason=(
                    "context_budget_exceeded"
                    if verified.generation_metadata.finish_reason == "context_budget"
                    else "empty_generation"
                    if not verified.raw_text.strip()
                    else "generation_incomplete"
                    if verified.generation_metadata.finish_reason not in {"stop", "eos_token"}
                    else "llm_refusal"
                    if verified.raw_text.strip() == REFUSAL
                    else "nli_verification_failed"
                )
                if verified.refused
                else None,
            )

    def close(self) -> None:
        """Attempt every cleanup even when another component fails to close."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors = []
            for component in (
                self._legacy_verifier,
                self._generator,
                self._fact_verifier,
                self._retriever,
            ):
                if component is not None:
                    try:
                        component.close()
                    except Exception as exc:
                        errors.append(str(exc))
            self.documents.clear()
            self._generator = None
            self._fact_verifier = None
            self._retriever = None
            self._legacy_verifier = None
            self._parser = None
            if errors:
                raise PipelineError("Cleanup failed: " + "; ".join(errors))

    def __enter__(self) -> Self:
        self._ready()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
