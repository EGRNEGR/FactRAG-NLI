"""The legacy suite explicitly exercises development fallback implementations."""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def explicit_development_mode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Unit tests never load deployment weights or consume secrets from the real .env.
    monkeypatch.setenv("RAG_API_TOKENS", "[]")
    monkeypatch.setenv("RAG_LLM_API_BASE", "")
    monkeypatch.setenv("RAG_LLM_BACKEND", "llama_cpp")
    monkeypatch.setenv("RAG_LLM_TOKENIZER_MODE", "bytes")
    monkeypatch.setenv("RAG_GENERATION_REPAIR_ATTEMPTS", "0")
    monkeypatch.setenv("RAG_GENERATION_LANGUAGE", "ru")
    monkeypatch.setenv("RAG_NLI_EVIDENCE_FOCUS", "false")
    monkeypatch.setenv("RAG_RERANK_RRF_ALPHA", "0")
    monkeypatch.setenv("RAG_RERANK_SCORE_THRESHOLD", "0.5")
    for name in ("EMBEDDING", "RERANKER", "NLI", "LLM"):
        monkeypatch.setenv(f"RAG_{name}_MODEL_PATH", str(tmp_path / f"missing-{name}"))
    if request.path.name != "test_settings.py":
        monkeypatch.setenv("RAG_MODE", "development")
        monkeypatch.setenv("RAG_ALLOW_FALLBACK", "true")
        monkeypatch.setenv("RAG_QDRANT_PATH", str(tmp_path / "qdrant"))
