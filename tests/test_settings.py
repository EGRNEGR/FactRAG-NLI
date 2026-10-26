"""Configuration failures use artificial artifact layouts, never model inference."""

import json
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from settings import RAGSettings, validate_transformer_directory


def development(tmp_path: Path) -> RAGSettings:
    return RAGSettings(mode="development", allow_fallback=True, document_root=tmp_path)


def test_production_rejects_fallback() -> None:
    with pytest.raises(ValidationError, match="requires RAG_MODE"):
        RAGSettings(allow_fallback=True)


def test_production_requires_auth() -> None:
    with pytest.raises(ValidationError, match="API token"):
        RAGSettings()


def test_missing_models_fail_even_in_development(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="directory does not exist"):
        RAGSettings(mode="development", embedding_model_path=tmp_path / "absent")


@pytest.mark.parametrize("name", ["../outside.txt", "nested/../../outside.txt", "evil.exe"])
def test_document_boundaries(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        development(tmp_path).resolve_document(name, must_exist=False)


def test_empty_file_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").touch()
    with pytest.raises(ValueError, match="empty"):
        development(tmp_path).resolve_document("empty.txt")


def test_secret_not_in_repr_and_token_compare() -> None:
    secret = "a" * 32
    settings = RAGSettings(mode="development", allow_fallback=True, api_tokens=(SecretStr(secret),))
    assert secret not in repr(settings)
    assert settings.accepts_api_token(secret)
    assert not settings.accepts_api_token("неверный")


def test_shard_traversal_and_missing_shard(tmp_path: Path) -> None:
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="escapes"):
        validate_transformer_directory(tmp_path)
    index.write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="Missing or empty"):
        validate_transformer_directory(tmp_path)


def test_environment_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODE", "development")
    monkeypatch.setenv("RAG_ALLOW_FALLBACK", "true")
    monkeypatch.setenv("RAG_DENSE_CANDIDATES", "77")
    assert RAGSettings().dense_candidates == 77


def test_compatibility_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENVIRONMENT", "development")
    monkeypatch.setenv("RAG_ALLOW_FALLBACK", "true")
    monkeypatch.setenv("RAG_QDRANT_STORAGE_PATH", str(tmp_path))
    settings = RAGSettings()
    assert settings.environment == settings.mode == "development"
    assert settings.qdrant_storage_path == settings.qdrant_path == tmp_path


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_threshold(value: float) -> None:
    with pytest.raises(ValidationError):
        RAGSettings(mode="development", allow_fallback=True, nli_entailment_threshold=value)
