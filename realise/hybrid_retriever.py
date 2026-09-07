"""Disk-only neural retrieval with persistent Qdrant and recoverable BM25.

Qdrant is authoritative. The JSON BM25 cache is disposable and rebuilt on restart
when absent, stale or corrupt. One process owns an embedded Qdrant directory.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, Self

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from document_parser import DocumentChunk, ParsedChunk
from settings import RAGSettings, validate_transformer_directory

LOGGER = logging.getLogger(__name__)
_STOP_WORDS = frozenset(
    "и в во на к ко с со из за от до по при для а но или что как это the a an and or of to in on for with is are".split()
)


class RetrievalError(RuntimeError):
    """A local retrieval dependency, index or model failed."""


def tokenize(text: str) -> list[str]:
    """Unicode normalization, multilingual words/numbers; retain negation and units."""
    words = re.findall(
        r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold().replace("ё", "е"), re.UNICODE
    )
    return [word for word in words if word not in _STOP_WORDS]


def text_with_context(chunk: DocumentChunk) -> str:
    """Enrich external chunks; do not duplicate a prefix already emitted by the parser."""
    prefix = f"Документ: {chunk.document_id}\nРаздел: {' > '.join(chunk.section_path)}\n"
    return chunk.text if chunk.text.startswith(prefix) else prefix + chunk.text


def heading_only(chunk: DocumentChunk) -> bool:
    """An exact title without body is navigation, not evidence for a factual claim."""
    if not chunk.section_path:
        return False
    body = re.sub(r"^Документ:[^\n]*\nРаздел:[^\n]*\n", "", chunk.text).strip()

    def normalize(value: str) -> str:
        value = re.sub(r"^[\d.]+\s+", "", value.strip())
        return " ".join(value.casefold().split())

    return bool(body) and normalize(body) == normalize(chunk.section_path[-1])


class Encoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one finite vector per input."""
        ...


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one probability in [0, 1] per pair."""
        ...


class HashingEncoder:
    """Explicit development-only lexical embeddings, never a neural substitute."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 32:
            raise ValueError("dimension must be >= 32")
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vector = [0.0] * self.dimension
            for word in tokenize(text):
                slot = (
                    int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
                    % self.dimension
                )
                vector[slot] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            result.append([x / norm for x in vector])
        return result


class LexicalReranker:
    """Explicit development-only query coverage."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        terms = set(tokenize(query))
        return [len(terms.intersection(tokenize(text))) / max(len(terms), 1) for text in texts]


class SentenceTransformerEncoder:
    """BGE-M3 loaded from an absolute local directory with no remote code."""

    def __init__(self, settings: RAGSettings) -> None:
        settings.activate_offline_mode()
        validate_transformer_directory(settings.embedding_model_path)
        if not (settings.embedding_model_path / "modules.json").is_file():
            raise RetrievalError(
                "BGE-M3 requires its SentenceTransformer modules.json; implicit mean pooling is forbidden"
            )
        from sentence_transformers import SentenceTransformer
        from model_runtime import runtime

        device, dtype = runtime(settings)

        tokenizer_options = {
            "processor_kwargs"
            if "processor_kwargs" in inspect.signature(SentenceTransformer).parameters
            else "tokenizer_kwargs": {"local_files_only": True}
        }

        self._model = SentenceTransformer(
            str(settings.embedding_model_path),
            device=device,
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={"local_files_only": True},
            **tokenizer_options,
        )
        if device.startswith("cuda"):
            self._model.to(dtype=dtype)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise RetrievalError("Embedding model does not declare a positive dimension")
        self.dimension = int(dimension)
        self._batch_size = settings.embedding_batch_size

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch

        with torch.inference_mode():
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [[float(value) for value in vector] for vector in vectors]


class CrossEncoderReranker:
    """Single-logit bge-reranker with explicit sigmoid normalization."""

    def __init__(self, settings: RAGSettings) -> None:
        settings.activate_offline_mode()
        validate_transformer_directory(settings.reranker_model_path)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from model_runtime import runtime

        device, dtype = runtime(settings)

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(settings.reranker_model_path), local_files_only=True, trust_remote_code=False
        )
        self._model, loading = AutoModelForSequenceClassification.from_pretrained(
            str(settings.reranker_model_path),
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        if any(loading.get(name) for name in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RetrievalError(
                "Incomplete reranker checkpoint; randomly initialized parameters are forbidden"
            )
        if self._model.config.num_labels != 1:
            raise RetrievalError("Reranker must expose exactly one classification logit")
        self._model.to(device=device, dtype=dtype).eval()
        self._device = device
        self._max_length = min(settings.rerank_max_length, int(self._tokenizer.model_max_length))
        self._batch_size = settings.rerank_batch_size
        self._window_tokens = settings.rerank_window_tokens
        self.diagnostics: list[dict[str, int]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        import torch

        scores: list[float] = []
        self.diagnostics = []
        query_length = len(self._tokenizer.encode(query, add_special_tokens=False))
        pair_limit = min(self._max_length, query_length + self._window_tokens + 4)
        if pair_limit - query_length - 4 <= 64:
            raise RetrievalError("Query leaves insufficient reranker context")
        with torch.inference_mode():
            for offset in range(0, len(texts), self._batch_size):
                batch = [
                    re.sub(r"^Документ:[^\n]*\nРаздел:[^\n]*\n", "", text)
                    for text in texts[offset : offset + self._batch_size]
                ]
                encoded = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="only_second",
                    max_length=pair_limit,
                    stride=64,
                    return_overflowing_tokens=True,
                    return_tensors="pt",
                )
                mapping = encoded.pop("overflow_to_sample_mapping").tolist()
                values: list[float] = []
                for start in range(0, len(mapping), self._batch_size):
                    inputs = {
                        key: value[start : start + self._batch_size].to(self._device)
                        for key, value in encoded.items()
                    }
                    logits = self._model(**inputs).logits.reshape(-1)
                    values.extend(float(v) for v in torch.sigmoid(logits.float()).cpu().tolist())
                for index, text in enumerate(batch):
                    windows = [
                        v for owner, v in zip(mapping, values, strict=True) if owner == index
                    ]
                    scores.append(max(windows))
                    self.diagnostics.append(
                        {
                            "pair_tokens": query_length
                            + 4
                            + len(self._tokenizer.encode(text, add_special_tokens=False)),
                            "windows": len(windows),
                            "window_limit": pair_limit,
                        }
                    )
        return scores


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Legacy score is the RRF score; reranking probability is separate."""

    chunk: ParsedChunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    sources: tuple[str, ...] = ()
    final_score: float | None = None

    @property
    def rrf_score(self) -> float:
        return self.score


@dataclass(slots=True, frozen=True)
class RetrieverConfig:
    """Legacy constructor adapter; values are revalidated through RAGSettings."""

    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    dense_candidates: int = 60
    sparse_candidates: int = 60
    rerank_candidates: int = 60
    top_k: int = 5
    rerank_threshold: float = 0.5


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """One-based weighted RRF; duplicate IDs are invalid, ties sort by chunk ID."""
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    for name in sorted(rankings):
        ranking = rankings[name]
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"Duplicate IDs in {name} ranking")
        weight = (weights or {}).get(name, 1.0)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("RRF weights must be finite and nonnegative")
        if weight == 0:
            continue
        for rank, identifier in enumerate(ranking, 1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + rank)
    return dict(sorted(fused.items(), key=lambda item: (-item[1], item[0])))


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    document_id: str
    text: str
    source: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    element_ids: tuple[str, ...]
    token_count: int
    metadata: dict[str, Any]
    table_context: Any
    footnote_context: Any
    embedding_signature: str

    def chunk(self) -> DocumentChunk:
        return DocumentChunk(
            self.chunk_id,
            self.text,
            self.document_id,
            self.source,
            self.section_path,
            self.page_start,
            self.page_end,
            self.element_ids,
            self.token_count,
            {
                **self.metadata,
                "table_context": self.table_context,
                "footnote_context": self.footnote_context,
            },
        )


def document_complete(chunks: Sequence[ParsedChunk]) -> bool:
    """Legacy records are complete; managed documents require every manifest ordinal."""
    if not chunks:
        return False
    counts = [c.metadata.get("ingestion_chunk_count") for c in chunks]
    if all(n is None for n in counts):
        return True
    expected = counts[0]
    return (
        type(expected) is int
        and expected == len(chunks)
        and all(n == expected for n in counts)
        and {c.metadata.get("ingestion_ordinal") for c in chunks} == set(range(expected))
        and all(c.metadata.get("file_sha256") == c.document_id for c in chunks)
    )


class HybridRetriever:
    """Serialize model/index access; fail closed after ambiguous storage failures."""

    def __init__(
        self,
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        *,
        settings: RAGSettings | None = None,
        config: RetrieverConfig | None = None,
        qdrant_path: str | Path | None = None,
        qdrant_collection: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._failed = False
        self._client: QdrantClient | None = None
        try:
            current = settings or RAGSettings()
            values = current.model_dump()
            if config is not None:
                values.update(
                    dense_weight=config.dense_weight,
                    sparse_weight=config.sparse_weight,
                    rrf_k=config.rrf_k,
                    dense_candidates=config.dense_candidates,
                    sparse_candidates=config.sparse_candidates,
                    rerank_candidates=config.rerank_candidates,
                    rerank_top_k=config.top_k,
                    rerank_score_threshold=config.rerank_threshold,
                )
            if qdrant_path is not None:
                values["qdrant_path"] = Path(qdrant_path)
            if qdrant_collection is not None:
                values["qdrant_collection"] = qdrant_collection
            self.settings = RAGSettings.model_validate(values)
            self.settings.activate_offline_mode()
            if not self.settings.allow_fallback and (encoder is not None or reranker is not None):
                raise RetrievalError(
                    "Injected adapters are allowed only in explicit development fallback mode"
                )
            self.fallback_reasons: list[str] = []
            if encoder is None:
                try:
                    encoder = SentenceTransformerEncoder(self.settings)
                except Exception as exc:
                    if not self.settings.allow_fallback:
                        raise
                    self.fallback_reasons.append(f"embedding: {type(exc).__name__}: {exc}")
                    encoder = HashingEncoder(self.settings.fallback_dimension)
            if reranker is None:
                try:
                    reranker = CrossEncoderReranker(self.settings)
                except Exception as exc:
                    if not self.settings.allow_fallback:
                        raise
                    self.fallback_reasons.append(f"reranker: {type(exc).__name__}: {exc}")
                    reranker = LexicalReranker()
            self.encoder, self.reranker = encoder, reranker
            if self.fallback_reasons:
                LOGGER.warning("Explicit development fallbacks: %s", self.fallback_reasons)
            self._signature = self._embedding_signature()
            self._collection = self.settings.qdrant_collection
            self.settings.qdrant_storage_path.mkdir(parents=True, exist_ok=True)
            self._cache = self.settings.qdrant_storage_path / f"{self._collection}.bm25.json"
            self._client = QdrantClient(path=str(self.settings.qdrant_storage_path))
            if not self._client.collection_exists(self._collection):
                self._client.create_collection(
                    self._collection,
                    vectors_config=models.VectorParams(
                        size=self.encoder.dimension, distance=models.Distance.COSINE
                    ),
                )
            vector_config = self._client.get_collection(self._collection).config.params.vectors
            if (
                not isinstance(vector_config, models.VectorParams)
                or vector_config.size != self.encoder.dimension
                or vector_config.distance != models.Distance.COSINE
            ):
                raise RetrievalError(
                    "Collection vector dimension/distance mismatch; reindex in a new collection"
                )
            self._chunks: dict[str, DocumentChunk] = {}
            offset: Any = None
            while True:
                points, offset = self._client.scroll(
                    self._collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = _Payload.model_validate(point.payload)
                    if payload.embedding_signature != self._signature:
                        raise RetrievalError(
                            "Embedding signature changed; use a new collection and reindex"
                        )
                    self._chunks[payload.chunk_id] = payload.chunk()
                if offset is None:
                    break
            self._bm25: Any = None
            self._restore_bm25()
        except Exception as exc:
            if self._client is not None:
                self._client.close()
            raise RetrievalError(f"Retriever initialization failed: {exc}") from exc

    def _embedding_signature(self) -> str:
        digest = hashlib.sha256(
            f"context-v1:{type(self.encoder).__module__}.{type(self.encoder).__qualname__}:{self.encoder.dimension}".encode()
        )
        if isinstance(self.encoder, SentenceTransformerEncoder):
            for path in sorted(self.settings.embedding_model_path.rglob("*")):
                if path.is_file() and path.suffix in {
                    ".json",
                    ".safetensors",
                    ".bin",
                    ".model",
                    ".txt",
                }:
                    digest.update(
                        path.relative_to(self.settings.embedding_model_path).as_posix().encode()
                    )
                    with path.open("rb") as stream:
                        digest.update(hashlib.file_digest(stream, "sha256").digest())
        return digest.hexdigest()

    def _payload(self, chunk: DocumentChunk) -> dict[str, Any]:
        payload = _Payload(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            source=chunk.source,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            element_ids=chunk.element_ids,
            token_count=chunk.token_count,
            metadata=dict(chunk.metadata),
            table_context=chunk.metadata.get("table_context", ()),
            footnote_context=chunk.metadata.get("footnote_context", ()),
            embedding_signature=self._signature,
        )
        result: dict[str, Any] = json.loads(
            json.dumps(payload.model_dump(mode="json"), allow_nan=False)
        )
        return result

    def _restore_bm25(self) -> None:
        grouped: dict[str, list[ParsedChunk]] = {}
        for chunk in self._chunks.values():
            grouped.setdefault(chunk.document_id, []).append(chunk)
        self._pending_documents = {
            key for key, group in grouped.items() if not document_complete(group)
        }
        self._ordered_ids = sorted(
            key
            for key, chunk in self._chunks.items()
            if chunk.document_id not in self._pending_documents
        )
        corpus = [tokenize(text_with_context(self._chunks[key])) for key in self._ordered_ids]
        state = {"version": 1, "ids": self._ordered_ids, "tokens": corpus}
        # Exact comparison detects stale or manually corrupted cache contents.
        try:
            cached = json.loads(self._cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        self._bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None
        self._token_sets = [set(tokens) for tokens in corpus]
        if cached != state:
            self._write_cache(state)

    def _write_cache(self, state: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._cache.parent,
                prefix=self._cache.name,
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(state, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _ready(self) -> QdrantClient:
        if self._closed or self._failed or self._client is None:
            raise RetrievalError("Retriever is closed or failed; reopen to recover from Qdrant")
        return self._client

    def _vectors(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.encoder.encode(texts)
        if len(vectors) != len(texts) or any(
            len(v) != self.encoder.dimension or not all(math.isfinite(x) for x in v)
            for v in vectors
        ):
            raise RetrievalError(
                "Encoder returned invalid vector count, dimension or nonfinite values"
            )
        return vectors

    @property
    def size(self) -> int:
        with self._lock:
            self._ready()
            return len(self._chunks)

    def add(self, chunks: Sequence[ParsedChunk]) -> int:
        """Upsert by chunk ID; validate the whole batch before writing anything."""
        with self._lock:
            client = self._ready()
            if not chunks:
                return 0
            if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
                raise ValueError("Duplicate chunk IDs in batch")
            payloads = [self._payload(chunk) for chunk in chunks]
            vectors = self._vectors([text_with_context(chunk) for chunk in chunks])
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector,
                    payload=payload,
                )
                for chunk, vector, payload in zip(chunks, vectors, payloads, strict=True)
            ]
            try:
                client.upsert(self._collection, points=points, wait=True)
                self._chunks.update({chunk.chunk_id: chunk for chunk in chunks})
                self._restore_bm25()
            except Exception as exc:
                self._failed = True
                raise RetrievalError(
                    "Index mutation failed; restart to recover the BM25 cache"
                ) from exc
            return len(chunks)

    def get_chunk(self, chunk_id: str) -> ParsedChunk | None:
        """Read an indexed chunk without rerunning retrieval or changing its trace."""
        with self._lock:
            self._ready()
            return self._chunks.get(chunk_id)

    def document_chunks(self, document_id: str) -> tuple[ParsedChunk, ...]:
        """Read document records for content-hash deduplication and recovery."""
        with self._lock:
            self._ready()
            return tuple(c for c in self._chunks.values() if c.document_id == document_id)

    def add_incremental(self, chunks: Sequence[ParsedChunk], *, batch_size: int) -> int:
        """Bound embedding/upsert batches; rebuild BM25 once after the document.

        An interrupted write may leave a partial document in Qdrant. The ingestion
        manifest in each chunk detects this on retry; this instance fails closed.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with self._lock:
            client = self._ready()
            if not chunks:
                return 0
            if len({c.chunk_id for c in chunks}) != len(chunks):
                raise ValueError("Duplicate chunk IDs in document")
            payloads = [self._payload(c) for c in chunks]
            try:
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start : start + batch_size]
                    vectors = self._vectors([text_with_context(c) for c in batch])
                    points = [
                        models.PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                            vector=v,
                            payload=p,
                        )
                        for c, v, p in zip(
                            batch, vectors, payloads[start : start + batch_size], strict=True
                        )
                    ]
                    client.upsert(self._collection, points=points, wait=True)
                    self._chunks.update({c.chunk_id: c for c in batch})
                self._restore_bm25()
            except Exception as exc:
                self._failed = True
                raise RetrievalError(
                    "Incremental write failed; reopen and retry the file to recover"
                ) from exc
            return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Delete from both indexes; survive a restart between dense and cache writes."""
        with self._lock:
            client = self._ready()
            keys = [key for key, chunk in self._chunks.items() if chunk.document_id == document_id]
            if not keys:
                return 0
            try:
                client.delete(
                    self._collection,
                    points_selector=models.PointIdsList(
                        points=[str(uuid.uuid5(uuid.NAMESPACE_URL, key)) for key in keys]
                    ),
                    wait=True,
                )
                for key in keys:
                    del self._chunks[key]
                self._restore_bm25()
            except Exception as exc:
                self._failed = True
                raise RetrievalError(
                    "Delete failed; reopen the retriever before further queries"
                ) from exc
            return len(keys)

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        """Return reranked candidates; dense-only matches need no lexical overlap."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        limit = self.settings.rerank_top_k if top_k is None else top_k
        if limit < 1 or limit > self.settings.rerank_candidates:
            raise ValueError("top_k must be positive and within rerank_candidates")
        with self._lock:
            client = self._ready()
            self._hybrid_candidate_ids: list[str] = []
            self._retrieval_diagnostics: list[dict[str, Any]] = []
            if not self._chunks:
                return []
            hits = client.query_points(
                self._collection,
                query=self._vectors([query])[0],
                limit=self.settings.dense_candidates,
                with_payload=True,
                query_filter=models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchAny(any=sorted(self._pending_documents)),
                        )
                    ]
                )
                if self._pending_documents
                else None,
            ).points
            dense = {str(hit.payload["chunk_id"]): float(hit.score) for hit in hits if hit.payload}
            dense = dict(sorted(dense.items(), key=lambda item: (-item[1], item[0])))
            terms = tokenize(query)
            scores = self._bm25.get_scores(terms) if self._bm25 is not None else []
            # Okapi IDF can be zero or negative in tiny corpora; overlap, not score > 0,
            # determines whether the sparse candidate actually matches.
            sparse = (
                {
                    key: float(score)
                    for key, score, tokens in zip(
                        self._ordered_ids, scores, self._token_sets, strict=True
                    )
                    if tokens.intersection(terms)
                }
                if len(scores)
                else {}
            )
            sparse = dict(
                sorted(sparse.items(), key=lambda item: (-item[1], item[0]))[
                    : self.settings.sparse_candidates
                ]
            )
            fused = reciprocal_rank_fusion(
                {"dense": list(dense), "sparse": list(sparse)},
                weights={
                    "dense": self.settings.dense_weight,
                    "sparse": self.settings.sparse_weight,
                },
                k=self.settings.rrf_k,
            )
            candidates = list(fused)[: self.settings.rerank_candidates]
            self._hybrid_candidate_ids = candidates.copy()
            if not candidates:
                return []
            rerank_started = time.perf_counter()
            reranked = self.reranker.score(
                query, [text_with_context(self._chunks[key]) for key in candidates]
            )
            self._last_rerank_ms = (time.perf_counter() - rerank_started) * 1000
            if len(reranked) != len(candidates) or any(
                not math.isfinite(s) or not 0 <= s <= 1 for s in reranked
            ):
                raise RetrievalError("Reranker must return one finite probability per candidate")
            substantive = {key: not heading_only(self._chunks[key]) for key in candidates}
            result = [
                SearchResult(
                    self._chunks[key],
                    fused[key],
                    dense.get(key),
                    sparse.get(key),
                    score,
                    tuple(
                        name
                        for name, mapping in (("dense", dense), ("sparse", sparse))
                        if key in mapping
                    ),
                    self.settings.rerank_rrf_alpha
                    * fused[key]
                    / (
                        (self.settings.dense_weight + self.settings.sparse_weight)
                        / (self.settings.rrf_k + 1)
                    )
                    + (1 - self.settings.rerank_rrf_alpha) * score,
                )
                for key, score in zip(candidates, reranked, strict=True)
                if score >= self.settings.rerank_score_threshold and substantive[key]
            ]
            details = getattr(self.reranker, "diagnostics", [])
            self._retrieval_diagnostics = [
                {
                    "chunk_id": key,
                    "rrf_score": fused[key],
                    "rerank_score": score,
                    "eligible": score >= self.settings.rerank_score_threshold and substantive[key],
                    "heading_only": not substantive[key],
                    **(details[i] if i < len(details) else {}),
                }
                for i, (key, score) in enumerate(zip(candidates, reranked, strict=True))
            ]
            return sorted(
                result,
                key=lambda hit: (-float(hit.final_score or 0), -hit.score, hit.chunk.chunk_id),
            )[:limit]

    def search_timed(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[list[SearchResult], dict[str, float]]:
        """Return per-call timings while holding the index lock across measurement."""
        with self._lock:
            self._last_rerank_ms = 0.0
            started = time.perf_counter()
            results = self.search(query, top_k=top_k)
            elapsed = (time.perf_counter() - started) * 1000
            return results, {
                "search_ms": max(0.0, elapsed - self._last_rerank_ms),
                "rerank_ms": self._last_rerank_ms,
            }

    def search_trace(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[list[SearchResult], dict[str, float], list[str], list[dict[str, Any]]]:
        """Capture pre-rerank candidates from the same retrieval call atomically."""
        with self._lock:
            results, timing = self.search_timed(query, top_k=top_k)
            return (
                results,
                timing,
                self._hybrid_candidate_ids.copy(),
                self._retrieval_diagnostics.copy(),
            )

    def close(self) -> None:
        """Release the embedded storage lock; safe to call repeatedly."""
        with self._lock:
            if not self._closed and self._client is not None:
                self._client.close()
            self._closed = True

    def __enter__(self) -> Self:
        self._ready()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
