"""Calibration contracts independent of the fifteen development questions."""

from __future__ import annotations

import httpx
import pytest
from types import SimpleNamespace
from typing import Any
from pathlib import Path
import json

from hybrid_retriever import CrossEncoderReranker, HashingEncoder, HybridRetriever, LexicalReranker
from pipeline import RAGPipeline

from generator_verifier import (
    GenerationMetadata,
    NLIFactVerifier,
    NLIProbabilities,
    RawGenerationResult,
    SourceContext,
    _HTTPBackend,
    ChatReply,
    HeuristicNLI,
    LocalLLMGenerator,
    evidence_windows,
    generation_system_prompt,
)
from settings import RAGSettings


class FixedNLI:
    def __init__(self, entailment: float, contradiction: float) -> None:
        self.result = NLIProbabilities(
            p_entailment=entailment,
            p_contradiction=contradiction,
            p_neutral=1 - entailment - contradiction,
        )

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        return self.result


@pytest.mark.parametrize(
    "entailment,contradiction,accepted",
    [
        (0.41, 0.19, True),
        (0.4, 0.19, False),
        (0.41, 0.2, False),
        (0.01, 0.01, False),
    ],
)
def test_strict_joint_nli_rule(entailment: float, contradiction: float, accepted: bool) -> None:
    verifier = NLIFactVerifier(RAGSettings(), backend=FixedNLI(entailment, contradiction))
    result = verifier.verify(
        RawGenerationResult(
            text="Факт [S1].",
            contexts={
                "S1": SourceContext(
                    chunk_id="a", document_id="doc", text="Факт.", citation="source"
                )
            },
            metadata=GenerationMetadata(backend="test", finish_reason="stop"),
        )
    )
    assert (not result.refused) is accepted
    verifier.close()


def test_local_tokenizer_replaces_byte_estimate() -> None:
    settings = RAGSettings(
        llm_api_base="http://127.0.0.1:8080/v1", llm_tokenizer_mode="llama_server"
    )
    backend = _HTTPBackend(settings)
    backend.client.close()

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tokenize"
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    backend.client = httpx.Client(
        base_url="http://127.0.0.1:8080/v1/", transport=httpx.MockTransport(handle)
    )
    try:
        assert backend.count_tokens("Многоязычная строка") == 3
    finally:
        backend.close()


def test_reranker_scores_tail_overflow_window() -> None:
    import torch

    class Tokenizer:
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return list(range(2 if text == "query" else 600))

        def __call__(self, queries: list[str], texts: list[str], **kwargs: Any) -> dict[str, Any]:
            assert queries == ["query"] and texts == ["body with relevant tail"]
            assert kwargs["truncation"] == "only_second"
            assert kwargs["return_overflowing_tokens"] is True
            return {
                "input_ids": torch.tensor([[0], [1]]),
                "overflow_to_sample_mapping": torch.tensor([0, 0]),
            }

    class Model:
        def __call__(self, **inputs: Any) -> Any:
            return SimpleNamespace(logits=inputs["input_ids"].float() * 16 - 8)

    scorer = CrossEncoderReranker.__new__(CrossEncoderReranker)
    scorer._tokenizer, scorer._model = Tokenizer(), Model()
    scorer._device, scorer._max_length, scorer._batch_size = "cpu", 512, 1
    scorer._window_tokens = 384
    assert scorer.score("query", ["body with relevant tail"])[0] > 0.99
    assert scorer.diagnostics[0]["windows"] == 2
    assert scorer.diagnostics[0]["pair_tokens"] > 512


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_repair_is_bounded_and_reverified(tmp_path: Path, repair_succeeds: bool) -> None:
    class Chat:
        calls = 0

        def count_tokens(self, text: str) -> int:
            return len(text.split())

        def complete(self, system_prompt: str, prompt: str) -> ChatReply:
            self.calls += 1
            if self.calls == 2:
                assert json.loads(prompt)["revision"]["rejected_claims"]
            value = 10 if repair_succeeds and self.calls == 2 else 99
            return ChatReply(text=f"Давление составляет {value} МПа [S1].", finish_reason="stop")

        def close(self) -> None:
            pass

    settings = RAGSettings(
        document_root=tmp_path,
        qdrant_path=tmp_path / "db",
        generation_repair_attempts=1,
        rerank_score_threshold=0.1,
    )
    chat = Chat()
    with RAGPipeline(
        settings,
        retriever=HybridRetriever(HashingEncoder(64), LexicalReranker(), settings=settings),
        generator=LocalLLMGenerator(settings, backend=chat),
        fact_verifier=NLIFactVerifier(settings, backend=HeuristicNLI()),
    ) as pipeline:
        pipeline.ingest_text("Давление составляет 10 МПа.")
        response = pipeline.query("Давление?")
        assert chat.calls == 2 and len(response.verification_attempts) == 2
        assert response.refused is not repair_succeeds
        assert response.verification_attempts[0]["refused"]
        assert response.execution_stats["nli_ms"] == sum(
            a["verification_ms"] for a in response.verification_attempts
        )


def test_evidence_windows_preserve_adjacent_exception() -> None:
    text = "Pressure is 10 MPa. However, the emergency limit is 5 MPa. Temperature is 20 C."
    windows = evidence_windows(text)
    assert all("emergency limit" in window for window in windows if "Pressure is 10" in window)
    assert "Temperature is 20 C." in windows
    assert evidence_windows("Single sentence.") == ["Single sentence."]


def test_source_language_keeps_refusal_and_citation_contract() -> None:
    from generator_verifier import REFUSAL, SYSTEM_PROMPT

    assert generation_system_prompt(RAGSettings(generation_language="ru")) == SYSTEM_PROMPT
    prompt = generation_system_prompt(RAGSettings(generation_language="source"))
    assert "Отвечай по-русски" not in prompt
    assert "не переводи" in prompt
    assert REFUSAL in prompt and "[S1]" in prompt
