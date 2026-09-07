"""Мы проверяем владение моделями, локальные загрузки и изоляцию NLI-политик."""

from pathlib import Path
import threading

import pytest

from document_parser import DocumentChunk
from document_processor import IngestionResult
from generator_verifier import TransformersNLI
from hybrid_retriever import HashingEncoder, HybridRetriever, LexicalReranker
from settings import RAGSettings
from summarization import _UnfocusedNLI
from web_bridge import WebBridge, get_bridge, safe_filename
import web_bridge


def config(tmp_path: Path) -> RAGSettings:
    return RAGSettings(
        mode="development",
        allow_fallback=True,
        document_root=tmp_path / "documents",
        qdrant_path=tmp_path / "index",
    )


def test_upload_paths_sizes_and_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[Path] = []

    class Processor:
        def add(self, path: Path, retriever: HybridRetriever) -> IngestionResult:
            paths.append(path)
            return IngestionResult(
                str(path), path.parent.name, "duplicate" if len(paths) > 1 else "added", 2
            )

    monkeypatch.setattr(web_bridge, "DocumentProcessor", lambda _: Processor())
    settings = config(tmp_path)
    bridge = WebBridge(settings)
    bridge._qa._retriever = HybridRetriever(
        HashingEncoder(32), LexicalReranker(), settings=settings
    )
    try:
        assert "добавили" in bridge.ingest("../../manual.txt", b"Our content", lambda _: None)
        assert "дубликат" in bridge.ingest("manual.txt", b"Our content", lambda _: None)
        assert paths[0] == paths[1]
        assert paths[0].is_relative_to(settings.document_root / "uploads")
        assert paths[0].read_bytes() == b"Our content"
        with pytest.raises(ValueError):
            bridge.ingest("manual.exe", b"Our content", lambda _: None)
        with pytest.raises(ValueError):
            bridge.ingest("manual.txt", b"", lambda _: None)
        monkeypatch.setattr(web_bridge, "MAX_UPLOAD_BYTES", 2)
        with pytest.raises(ValueError):
            bridge.ingest("manual.txt", b"long", lambda _: None)
    finally:
        bridge.close()


def test_document_catalog_uses_existing_owner(tmp_path: Path) -> None:
    settings = config(tmp_path)
    with HybridRetriever(HashingEncoder(32), LexicalReranker(), settings=settings) as retriever:
        retriever.add(
            [DocumentChunk("c1", "Our data", "doc", "manual.txt", (), None, None, (), 2, {})]
        )
    bridge = WebBridge(settings)
    try:
        assert bridge._qa._retriever is None
        assert bridge.documents()[0].document_id == "doc"
        assert bridge._qa._retriever is None
        bridge._qa._retriever = HybridRetriever(
            HashingEncoder(32), LexicalReranker(), settings=settings
        )
        assert bridge.documents()[0].chunks == 1
    finally:
        bridge.close()
    with pytest.raises(RuntimeError):
        bridge.documents()


def test_summary_restores_shared_qa_focus_even_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = object.__new__(TransformersNLI)
    encoder = HashingEncoder(32)
    backend.evidence_encoder = encoder

    def predict(self: TransformersNLI, premise: str, hypothesis: str) -> None:
        assert self.evidence_encoder is None
        raise RuntimeError("Our simulated inference error")

    monkeypatch.setattr(TransformersNLI, "predict", predict)
    adapter = _UnfocusedNLI(backend, threading.RLock())
    with pytest.raises(RuntimeError):
        adapter.predict("Our evidence", "Our claim")
    assert backend.evidence_encoder is encoder


def test_cached_owner_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    owner = WebBridge(config(tmp_path))
    monkeypatch.setattr(web_bridge, "WebBridge", lambda: owner)
    get_bridge.clear()
    try:
        assert get_bridge() is get_bridge() is owner
    finally:
        owner.close()
        get_bridge.clear()


def test_windows_filename_is_not_device_or_ads() -> None:
    assert safe_filename(r"C:\private\CON.txt") == "document_CON.txt"
    assert ":" not in safe_filename("file:stream.txt")
    assert safe_filename("../../manual.md") == "manual.md"
