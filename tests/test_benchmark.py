"""GPU selection and benchmark accounting contracts; streaming uses a local mock."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from benchmark import percentiles, query_record, run
from generator_verifier import VerificationError, _HTTPBackend
from model_runtime import runtime
from pipeline import RAGPipeline
from settings import RAGSettings


def test_percentiles_include_extremes() -> None:
    assert percentiles([]) == {}
    assert percentiles([30, 10, 20]) == {"p50_ms": 20, "p95_ms": 30}


def test_benchmark_validation_does_not_create_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run(RAGSettings(), tmp_path / "unused", 0, 1)
    assert not (tmp_path / "unused").exists()


def test_query_error_is_measured() -> None:
    with RAGPipeline(RAGSettings()) as pipeline:
        result = query_record(pipeline, "")
    assert result["ok"] is False and result["wall_ms"] >= 0


def test_device_auto_and_explicit_cuda_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime(RAGSettings(model_device="auto")) == ("cpu", torch.float32)
    with pytest.raises(RuntimeError, match="CPU retry is forbidden"):
        runtime(RAGSettings(model_device="cuda"))


def test_real_cuda_math_when_available() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; GPU evidence must come from a CUDA host")
    device, dtype = runtime(RAGSettings())
    assert device == "cuda" and dtype in (torch.bfloat16, torch.float16)
    with torch.inference_mode():
        matrix = torch.randn(128, 128, device=device, dtype=dtype)
        result = matrix @ matrix
        assert bool(torch.isfinite(result).all().item())


@pytest.mark.parametrize("failure", [None, "empty", "incomplete", "error"])
def test_stream_ttft_usage_and_failure(failure: str | None) -> None:
    settings = RAGSettings(
        llm_api_base="http://127.0.0.1:8080/v1", llm_api_model="test", llm_api_stream=True
    )
    backend = _HTTPBackend(settings)
    backend.client.close()

    def handle(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        events: list[dict[str, object]] = [
            {"choices": [{"delta": {"role": "assistant"}}]},
        ]
        if failure == "error":
            events.append({"error": {"message": "device out of memory"}})
        elif failure != "empty":
            events.append({"choices": [{"delta": {"content": "10 МПа [S1]."}}]})
        if failure != "incomplete":
            events.append(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 8},
                }
            )
        content = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(
            200, text=content + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )

    backend.client = httpx.Client(
        base_url=settings.llm_api_base + "/",  # type: ignore[operator]
        transport=httpx.MockTransport(handle),
    )
    try:
        if failure:
            with pytest.raises(VerificationError):
                backend.complete("system", "question")
        else:
            result = backend.complete("system", "question")
            assert result.ttft_ms is not None and result.ttft_ms >= 0
            assert result.prompt_tokens == 100 and result.completion_tokens == 8
            assert result.text == "10 МПа [S1]."
    finally:
        backend.close()
