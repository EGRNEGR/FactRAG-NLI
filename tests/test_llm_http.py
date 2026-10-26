"""HTTP backend contracts using httpx transports; never contact deployment services."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from generator_verifier import LocalLLMGenerator, VerificationError, _HTTPBackend
from settings import RAGSettings
from document_parser import DocumentChunk


def configured() -> RAGSettings:
    return RAGSettings(
        mode="development",
        allow_fallback=True,
        llm_backend="vllm",
        llm_api_base="http://127.0.0.1:8080/v1/",
    )


def test_dotenv_field_with_extra_forbid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LLM_API_BASE")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("RAG_LLM_API_BASE=http://127.0.0.1:8080/v1\n")
    assert RAGSettings().llm_api_base == "http://127.0.0.1:8080/v1"
    (tmp_path / ".env").write_text("RAG_LLM_API_BASE=http://127.0.0.1:8080/v1\nRAG_TYPO=1\n")
    with pytest.raises(ValidationError, match="Extra inputs"):
        RAGSettings()


@pytest.mark.parametrize(
    "url", ["https://example.com/v1", "http://127.0.0.1@evil.com", "http://127.0.0.1/v1?secret=x"]
)
def test_external_endpoints_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        RAGSettings(mode="development", allow_fallback=True, llm_api_base=url)


def test_http_completion_and_no_native_loader() -> None:
    generator = LocalLLMGenerator(configured())
    assert isinstance(generator._backend, _HTTPBackend)
    backend = generator._backend
    backend.client.close()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-qwen"}]})
        body = json.loads(request.content)
        assert body["model"] == "local-qwen"
        assert "[S1]" in body["messages"][0]["content"]
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "10 МПа [S1]."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 7},
            },
        )

    backend.client = httpx.Client(
        base_url="http://127.0.0.1:8080/v1/", transport=httpx.MockTransport(handle)
    )
    try:
        chunk = DocumentChunk("one", "10 МПа", "doc", "source.txt", (), None, None, (), 3)
        result = generator.generate("Давление?", [chunk])
        assert result.text == "10 МПа [S1]."
        assert result.metadata.completion_tokens == 7
        assert [r.url.path for r in requests] == ["/v1/models", "/v1/chat/completions"]
    finally:
        generator.close()


@pytest.mark.parametrize("failure", ["timeout", "invalid", "http_error", "redirect"])
def test_failures_raise_without_native_fallback(failure: str) -> None:
    settings = configured().model_copy(update={"llm_api_model": "qwen"})
    generator = LocalLLMGenerator(settings)
    assert isinstance(generator._backend, _HTTPBackend)
    backend = generator._backend
    backend.client.close()

    def handle(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "redirect":
            return httpx.Response(302, headers={"location": "https://example.com"})
        return httpx.Response(500 if failure == "http_error" else 200, json={})

    backend.client = httpx.Client(
        base_url="http://127.0.0.1/v1/", transport=httpx.MockTransport(handle)
    )
    try:
        with pytest.raises(VerificationError, match="Local generation failed"):
            generator.generate(
                "Давление?", [DocumentChunk("one", "10 МПа", "doc", "x", (), None, None, (), 3)]
            )
    finally:
        generator.close()


def test_api_base_overrides_native_backend() -> None:
    settings = configured().model_copy(update={"llm_backend": "llama_cpp"})
    generator = LocalLLMGenerator(settings)
    try:
        assert isinstance(generator._backend, _HTTPBackend)
    finally:
        generator.close()
