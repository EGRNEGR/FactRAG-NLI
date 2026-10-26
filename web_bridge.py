"""Мы разделяем локальные модели и сериализуем операции нашего Web-интерфейса."""

from __future__ import annotations

import atexit
import hashlib
import re
import threading
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import streamlit as st
from qdrant_client import QdrantClient

from document_parser import DocumentChunk
from document_processor import DocumentProcessor, SUPPORTED_SUFFIXES
from generator_verifier import NLIFactVerifier, TransformersNLI
from hybrid_retriever import _Payload, document_complete
from pipeline import RAGPipeline
from settings import RAGSettings
from summarization import SummarizationPipeline

Mode = Literal["qa", "summary"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class DocumentInfo:
    document_id: str
    name: str
    chunks: int


def safe_filename(name: str) -> str:
    # Мы исключаем пути, Windows ADS и зарезервированные имена устройств.
    base = Path(name.replace("\\", "/")).name
    suffix = Path(base).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Мы принимаем только TXT, MD, PDF и DOCX")
    stem = re.sub(r"[^\w.-]", "_", Path(base).stem).strip(". ")[:100] or "document"
    if stem.split(".")[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(10)),
        *(f"LPT{i}" for i in range(10)),
    }:
        stem = "document_" + stem
    return stem + suffix


class WebBridge:
    """Мы храним модели в общем ресурсе, а переписку — только в UI-сессии."""

    def __init__(self, settings: RAGSettings | None = None) -> None:
        configured = settings or RAGSettings()
        self.settings = RAGSettings.model_validate(configured.model_dump())
        self._lock = threading.RLock()
        self._qa = RAGPipeline(self.settings)
        self._summary: SummarizationPipeline | None = None
        self._processor: DocumentProcessor | None = None
        self._closed = False
        atexit.register(self.close)

    def _ready(self) -> None:
        if self._closed:
            raise RuntimeError("Мы уже закрыли наш Web-контур")

    def documents(self) -> list[DocumentInfo]:
        with self._lock:
            self._ready()
            chunks: list[DocumentChunk] = []
            if self._qa._retriever is not None:
                chunks = list(self._qa._retriever._chunks.values())
            elif self.settings.qdrant_storage_path.exists():
                with closing(QdrantClient(path=str(self.settings.qdrant_storage_path))) as client:
                    if client.collection_exists(self.settings.qdrant_collection):
                        offset: Any = None
                        while True:
                            points, offset = client.scroll(
                                self.settings.qdrant_collection,
                                limit=256,
                                offset=offset,
                                with_payload=True,
                                with_vectors=False,
                            )
                            chunks.extend(
                                _Payload.model_validate(p.payload).chunk() for p in points
                            )
                            if offset is None:
                                break
            grouped: dict[str, list[DocumentChunk]] = defaultdict(list)
            for chunk in chunks:
                grouped[chunk.document_id].append(chunk)
            return sorted(
                [
                    DocumentInfo(doc_id, Path(items[0].source).name, len(items))
                    for doc_id, items in grouped.items()
                    if document_complete(items)
                ],
                key=lambda doc: (doc.name.casefold(), doc.document_id),
            )

    def _summarizer(self) -> SummarizationPipeline:
        if self._summary is None:
            retriever = self._qa.retriever
            if self._qa._fact_verifier is None:
                self._qa._fact_verifier = NLIFactVerifier(
                    self.settings, evidence_encoder=retriever.encoder
                )
            backend = self._qa._fact_verifier.backend
            if not isinstance(backend, TransformersNLI):
                raise RuntimeError(
                    "Мы используем для Web-саммари только реальную локальную NLI-модель"
                )
            self._summary = SummarizationPipeline.from_shared_models(
                self.settings, retriever, backend, self._lock
            )
        return self._summary

    def run(
        self, question: str, mode: Mode, document_id: str | None, on_status: Callable[[str], None]
    ) -> dict[str, Any]:
        on_status("Мы ожидаем доступ к нашим моделям…")
        with self._lock:
            self._ready()
            if mode == "qa":
                on_status("Мы ищем контекст, формируем ответ и проверяем его утверждения…")
                response = self._qa.query(question)
                result = response.model_dump(mode="json")
                on_status("Мы завершили поиск и проверку фактов")
                # Мы не передаём непроверенный ответ в UI даже внутри скрытых полей.
                return {
                    "answer": response.answer,
                    "mode": mode,
                    "sources": result["sources"],
                    "claims": result["claims"],
                    "execution_stats": response.execution_stats,
                    "refused": response.refused,
                    "policy": "Мы проверяем утверждения по источникам; вероятностная NLI-проверка может ошибаться.",
                }
            if mode != "summary":
                raise ValueError("Мы не поддерживаем этот режим")
            if not document_id or not any(d.document_id == document_id for d in self.documents()):
                raise ValueError("Мы не нашли выбранный завершённый документ")
            # Мы разделяем PyTorch-модели; генератор работает в одном внешнем llama-server.
            if not self.settings.llm_api_base:
                raise RuntimeError(
                    "Мы запускаем Web-саммари через локальный llama-server: нужен RAG_LLM_API_BASE"
                )
            on_status(
                "Мы отбираем разнообразные фрагменты, составляем саммари и удаляем противоречия…"
            )
            summary = self._summarizer().summarize(document_id)
            on_status("Мы завершили NLI-проверку и удаление повторов")
            return {
                "answer": summary.summary,
                "mode": mode,
                "sources": [s.model_dump(mode="json") for s in summary.sources],
                "claims": [c.model_dump(mode="json") for c in summary.claims],
                "execution_stats": summary.execution_stats,
                "refused": summary.refused,
                "policy": "Мы обобщаем выборку фрагментов и допускаем NEUTRAL. Мы не гарантируем полноту документа и подтверждение всех фактов.",
            }

    def ingest(self, name: str, data: bytes, on_status: Callable[[str], None]) -> str:
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Мы принимаем непустой файл размером до 20 МБ")
        filename = safe_filename(name)
        on_status("Мы проверяем файл и вычисляем его SHA-256…")
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            self._ready()
            root = self.settings.document_root.resolve()
            folder = (root / "uploads" / digest).resolve()
            if not folder.is_relative_to(root):
                raise ValueError("Мы не сохраняем файл за пределами нашей базы")
            folder.mkdir(parents=True, exist_ok=True)
            target = (folder / filename).resolve()
            if not target.is_relative_to(folder):
                raise ValueError("Мы не следуем внешним ссылкам при загрузке")
            try:
                with target.open("xb") as stream:
                    stream.write(data)
            except FileExistsError:
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(
                        "Мы обнаружили несовпадение содержимого сохранённого файла"
                    ) from None
            on_status("Мы разбиваем документ и добавляем новые фрагменты в индекс…")
            if self._processor is None:
                self._processor = DocumentProcessor(self.settings)
            result = self._processor.add(target, self._qa.retriever)
            if result.status == "duplicate":
                return f"Мы уже добавляли {filename}; пропускаем дубликат. Фрагментов: {result.chunks}."
            return f"Мы добавили {filename} в нашу базу. Фрагментов: {result.chunks}."

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._summary is not None:
                    self._summary.close()
            finally:
                self._qa.close()
                self._summary = None
                self._processor = None
                self._closed = True


@st.cache_resource(show_spinner=False)
def get_bridge() -> WebBridge:
    """Мы кэшируем владельца BGE-M3, реранкера и NLI один раз на процесс."""
    return WebBridge()
