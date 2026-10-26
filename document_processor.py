"""Snapshot user files, split on sentence boundaries and append durable hash manifests."""

from __future__ import annotations

import hashlib
import re
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from qdrant_client import QdrantClient

from document_parser import (
    ChunkingConfig,
    DocumentChunk,
    DocumentElement,
    DocumentParseError,
    DocumentParser,
    LocalTokenizer,
    ParsedDocument,
    RegexTokenCounter,
    TokenCounter,
)
from hybrid_retriever import _Payload, document_complete
from settings import RAGSettings

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".pdf", ".docx"})
_SENTENCE = re.compile(r'(?<=[.!?。！？])\s+(?=[A-ZА-ЯЁ0-9«"“(])')


def clean_text(text: str) -> str:
    """Remove extraction controls without destroying Unicode letters or punctuation."""
    return "".join(
        c
        for c in unicodedata.normalize("NFC", text).replace("\ufffd", " ")
        if c in "\n\t" or unicodedata.category(c) not in {"Cc", "Cf"}
    ).strip()


class SentenceParser(DocumentParser):
    """Prefer paragraph boundaries; overlap complete sentences within each section.

    A sentence or table row too large for the hard budget is rejected rather than
    truncated. Sentence boundaries are conservative punctuation heuristics, not NLP.
    """

    chunk_limit: int | None = None

    def chunk(
        self, elements: Sequence[DocumentElement], *, document_id: str, source: str
    ) -> Iterator[DocumentChunk]:
        ordinal = 0
        pending: list[DocumentElement] = []
        fresh = False

        def content(items: Sequence[DocumentElement]) -> str:
            notes = list(
                dict.fromkeys(
                    clean_text(str(n))
                    for e in items
                    for n in e.metadata.get("footnote_context", ())
                )
            )
            return (
                f"Документ: {document_id}\nРаздел: {' > '.join(items[0].section_path)}\n"
                + "\n".join(e.text for e in items)
                + ("\n" + "\n".join(notes) if notes else "")
            )

        def emit(items: Sequence[DocumentElement]) -> DocumentChunk:
            nonlocal ordinal
            if self.chunk_limit is not None and ordinal >= self.chunk_limit:
                raise DocumentParseError("Document exceeds ingestion_max_chunks")
            text = content(items)
            count = self.tokenizer.count(text)
            if count > self.chunking.max_tokens:
                raise DocumentParseError(
                    "Sentence, table row or metadata exceeds ingestion token budget"
                )
            pages = [e.page for e in items if e.page is not None]
            result = DocumentChunk(
                hashlib.sha256(f"ingestion-v1:{document_id}:{ordinal}:{text}".encode()).hexdigest(),
                text,
                document_id,
                source,
                items[0].section_path,
                min(pages) if pages else None,
                max(pages) if pages else None,
                tuple(dict.fromkeys(e.element_id for e in items)),
                count,
                {
                    "kinds": sorted({e.kind for e in items}),
                    "table_context": tuple(e.metadata for e in items if e.kind == "table"),
                    "footnote_context": tuple(
                        dict.fromkeys(
                            clean_text(str(n))
                            for e in items
                            for n in e.metadata.get("footnote_context", ())
                        )
                    ),
                    "tokenizer": type(self.tokenizer).__name__,
                },
            )
            ordinal += 1
            return result

        def tail(items: Sequence[DocumentElement]) -> list[DocumentElement]:
            result: list[DocumentElement] = []
            for item in reversed(items):
                candidate = [item, *result]
                if (
                    self.tokenizer.count("\n".join(e.text for e in candidate))
                    > self.chunking.overlap_tokens
                ):
                    break
                result = candidate
            return result

        for original in elements:
            element = replace(
                original,
                text=clean_text(original.text),
                section_path=tuple(clean_text(s) for s in original.section_path),
            )
            if not element.text:
                continue
            if pending and (
                pending[0].section_path != element.section_path or element.kind == "table"
            ):
                if fresh:
                    yield emit(pending)
                pending, fresh = [], False
            if element.kind == "table":
                # Existing parser preserves complete rows and repeats the Markdown header.
                for block in super().chunk([element], document_id=document_id, source=source):
                    if self.chunk_limit is not None and ordinal >= self.chunk_limit:
                        raise DocumentParseError("Document exceeds ingestion_max_chunks")
                    if block.token_count > self.chunking.max_tokens:
                        raise DocumentParseError("Table row exceeds ingestion token budget")
                    yield replace(
                        block,
                        chunk_id=hashlib.sha256(
                            f"ingestion-v1:{document_id}:{ordinal}:{block.text}".encode()
                        ).hexdigest(),
                    )
                    ordinal += 1
                continue
            for sentence in _SENTENCE.split(element.text):
                unit = replace(element, text=sentence)
                if self.tokenizer.count(content([unit])) > self.chunking.max_tokens:
                    raise DocumentParseError("Single sentence exceeds ingestion token budget")
                if (
                    pending
                    and self.tokenizer.count(content([*pending, unit])) > self.chunking.max_tokens
                ):
                    if fresh:
                        yield emit(pending)
                    pending, fresh = tail(pending), False
                    while (
                        pending
                        and self.tokenizer.count(content([*pending, unit]))
                        > self.chunking.max_tokens
                    ):
                        pending.pop(0)
                pending.append(unit)
                fresh = True
            if pending and self.tokenizer.count(content(pending)) >= self.chunking.target_tokens:
                yield emit(pending)
                pending, fresh = tail(pending), False
        if fresh:
            yield emit(pending)


class IncrementalIndex(Protocol):
    def document_chunks(self, document_id: str) -> tuple[DocumentChunk, ...]: ...
    def add_incremental(self, chunks: Sequence[DocumentChunk], *, batch_size: int) -> int: ...
    def delete_document(self, document_id: str) -> int: ...


@dataclass(frozen=True)
class IngestionResult:
    source: str
    sha256: str
    status: str
    chunks: int


@dataclass(frozen=True)
class IndexStatus:
    documents: int
    chunks: int
    incomplete_documents: int
    stored_chunks: int


def index_status(settings: RAGSettings) -> IndexStatus:
    """Inspect embedded payloads without loading a tokenizer, encoder or reranker."""
    if not settings.qdrant_storage_path.exists():
        return IndexStatus(0, 0, 0, 0)
    client = QdrantClient(path=str(settings.qdrant_storage_path))
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return IndexStatus(0, 0, 0, 0)
        groups: dict[str, list[DocumentChunk]] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
            for point in points:
                chunk = _Payload.model_validate(point.payload).chunk()
                groups.setdefault(chunk.document_id, []).append(chunk)
            if offset is None:
                break
        complete = [items for items in groups.values() if document_complete(items)]
        return IndexStatus(
            len(complete),
            sum(map(len, complete)),
            len(groups) - len(complete),
            sum(len(items) for items in groups.values()),
        )
    finally:
        client.close()


class DocumentProcessor:
    def __init__(self, settings: RAGSettings, *, tokenizer: TokenCounter | None = None) -> None:
        self.settings = RAGSettings.model_validate(settings.model_dump())
        if tokenizer is not None and not self.settings.allow_fallback:
            raise ValueError("Injected token counters require explicit development fallback")
        self._tokenizer = tokenizer
        self._parser: SentenceParser | None = None

    @property
    def parser(self) -> SentenceParser:
        if self._parser is None:
            self.settings.activate_offline_mode()
            tokenizer = self._tokenizer or (
                RegexTokenCounter()
                if self.settings.allow_fallback
                else LocalTokenizer(self.settings.embedding_model_path)
            )
            self._parser = SentenceParser(
                ChunkingConfig(
                    self.settings.ingestion_target_tokens,
                    self.settings.ingestion_max_tokens,
                    self.settings.ingestion_overlap_tokens,
                ),
                tokenizer=tokenizer,
                max_pages=self.settings.max_document_pages,
                max_file_bytes=self.settings.max_upload_file_size_mb * 1024 * 1024,
            )
            self._parser.chunk_limit = self.settings.ingestion_max_chunks
        return self._parser

    def parse_docx(self, path: Path) -> ParsedDocument:
        """Мы сохраняем порядок абзацев, заголовки и таблицы штатным Word-адаптером."""
        if path.suffix.lower() != ".docx":
            raise DocumentParseError("Мы ожидаем документ DOCX")
        return self.parser.parse(path, backend="docx")

    def _parse(self, snapshot: Path, source: str, digest: str) -> ParsedDocument:
        parsed = (
            self.parse_docx(snapshot)
            if snapshot.suffix.lower() == ".docx"
            else self.parser.parse(snapshot)
        )
        if parsed.document_id != digest:
            raise DocumentParseError("File snapshot changed during parsing")
        total = len(parsed.chunks)
        if not 0 < total <= self.settings.ingestion_max_chunks:
            raise DocumentParseError("Document has no chunks or exceeds ingestion_max_chunks")
        chunks = tuple(
            replace(
                c,
                source=source,
                metadata={
                    **c.metadata,
                    "file_sha256": digest,
                    "ingestion_chunk_count": total,
                    "ingestion_ordinal": i,
                },
            )
            for i, c in enumerate(parsed.chunks)
        )
        return replace(
            parsed,
            source=source,
            chunks=chunks,
            elements=tuple(replace(e, source=source) for e in parsed.elements),
        )

    def add(self, path: Path, index: IncrementalIndex) -> IngestionResult:
        source = path.expanduser().resolve(strict=True)
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise DocumentParseError("Мы ожидаем файл TXT, MD, PDF или DOCX")
        maximum = self.settings.max_upload_file_size_mb * 1024 * 1024
        if not 0 < source.stat().st_size <= maximum:
            raise DocumentParseError("File is empty or exceeds the configured size limit")
        # Snapshot eliminates the hash/parse TOCTOU race, including externally edited files.
        with tempfile.TemporaryDirectory(prefix="rag-ingestion-") as temporary:
            snapshot = Path(temporary) / ("document" + source.suffix.lower())
            digest = hashlib.sha256()
            total = 0
            with source.open("rb") as reader, snapshot.open("wb") as writer:
                while block := reader.read(1024 * 1024):
                    total += len(block)
                    if total > maximum:
                        raise DocumentParseError("File grew beyond the size limit")
                    writer.write(block)
                    digest.update(block)
            if not total:
                raise DocumentParseError("File became empty while reading")
            sha = digest.hexdigest()
            existing = index.document_chunks(sha)
            if document_complete(existing):
                return IngestionResult(str(source), sha, "duplicate", len(existing))
            parsed = self._parse(snapshot, str(source), sha)
            if existing:
                # Only incomplete records with this exact content hash can be removed.
                index.delete_document(sha)
            count = index.add_incremental(
                parsed.chunks, batch_size=self.settings.ingestion_batch_size
            )
            if count != len(parsed.chunks):
                raise RuntimeError(
                    "Index did not acknowledge every document chunk; reopen and retry"
                )
            return IngestionResult(str(source), sha, "added", count)
