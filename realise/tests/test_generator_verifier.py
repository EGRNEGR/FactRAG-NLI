"""Deterministic contract tests; neural inference is covered by a separate local smoke run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from document_parser import DocumentChunk
from generator_verifier import (
    ChatReply,
    ClaimStatus,
    GenerationMetadata,
    LocalLLMGenerator,
    NLIFactVerifier,
    NLIProbabilities,
    RawGenerationResult,
    REFUSAL,
    SourceContext,
    VerificationError,
    _LlamaBackend,
    label_indices,
    segment_claims,
)
from settings import RAGSettings


def settings(**overrides: Any) -> RAGSettings:
    return RAGSettings.model_validate({"mode": "development", "allow_fallback": True, **overrides})


def source(text: str = "Давление составляет 10 МПа.", identifier: str = "one") -> SourceContext:
    return SourceContext(
        chunk_id=identifier, document_id="gost", text=text, citation="ГОСТ, п. 4.2"
    )


def raw(
    text: str, *, contexts: dict[str, SourceContext] | None = None, finish: str = "stop"
) -> RawGenerationResult:
    return RawGenerationResult(
        text=text,
        contexts=contexts if contexts is not None else {"S1": source()},
        metadata=GenerationMetadata(backend="test", finish_reason=finish),
    )


ENTAILMENT = NLIProbabilities(p_entailment=0.90, p_neutral=0.08, p_contradiction=0.02)
NEUTRAL = NLIProbabilities(p_entailment=0.10, p_neutral=0.85, p_contradiction=0.05)
CONTRADICTION = NLIProbabilities(p_entailment=0.02, p_neutral=0.08, p_contradiction=0.90)


class RecordedNLI:
    def __init__(self, probabilities: NLIProbabilities | None = ENTAILMENT) -> None:
        self.probabilities = probabilities
        self.calls: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities | None:
        self.calls.append((premise, hypothesis))
        return self.probabilities


class NumericalNLI:
    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        return ENTAILMENT if "10 МПа" in hypothesis else CONTRADICTION


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Давление 10,5 МПа [S1]. Температура 20 °C [S2].",
            ["Давление 10,5 МПа [S1].", "Температура 20 °C [S2]."],
        ),
        (
            "Давление 10 МПа. [S1] Температура 20 °C. [S2]",
            ["Давление 10 МПа. [S1]", "Температура 20 °C. [S2]"],
        ),
        ("См. п. 4.2.1: допуск 10.5 [S1].", ["См. п. 4.2.1: допуск 10.5 [S1]."]),
        ("- Факт [S1]\n- Другой факт [S2]", ["Факт [S1]", "Другой факт [S2]"]),
        ("Факт.\n[S1]\nДругой факт [S2].", ["Факт.\n[S1]", "Другой факт [S2]."]),
        ("Факт [S1]; другой факт [S2].", ["Факт [S1];", "другой факт [S2]."]),
    ],
)
def test_segmentation(text: str, expected: list[str]) -> None:
    assert segment_claims(text) == expected


def test_verified_and_json_roundtrip() -> None:
    backend = RecordedNLI()
    result = NLIFactVerifier(settings(), backend=backend).verify(raw("Давление 10 МПа [S1]."))
    assert result.claims[0].status == ClaimStatus.VERIFIED
    assert result.hallucination_rate == 0
    assert not result.refused
    assert backend.calls == [(source().text, "Давление 10 МПа .")]
    assert (
        json.loads(result.model_dump_json())["claims"][0]["evidence"][0]["probabilities"][
            "p_entailment"
        ]
        == 0.9
    )


@pytest.mark.parametrize(
    "probabilities,status",
    [(CONTRADICTION, ClaimStatus.CONTRADICTION), (NEUTRAL, ClaimStatus.UNVERIFIED)],
)
def test_block_unsupported(probabilities: NLIProbabilities, status: ClaimStatus) -> None:
    result = NLIFactVerifier(settings(), backend=RecordedNLI(probabilities)).verify(
        raw("Давление 99 МПа [S1].")
    )
    assert result.cleaned_text == REFUSAL and not result.is_reliable
    assert result.claims[0].status == status and result.hallucination_rate == 1


def test_exactly_half_keeps_verified_but_more_than_half_refuses() -> None:
    verifier = NLIFactVerifier(settings(), backend=NumericalNLI())
    result = verifier.verify(raw("Давление 10 МПа [S1]. Давление 99 МПа [S1]."))
    assert not result.refused and "99" not in result.cleaned_text
    assert result.hallucination_rate == 0.5 and not result.is_reliable
    rejected = verifier.verify(
        raw("Давление 10 МПа [S1]. Давление 99 МПа [S1]. Давление 88 МПа [S1].")
    )
    assert rejected.refused and rejected.cleaned_text == REFUSAL
    assert rejected.hallucination_rate == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("Факт.", "missing_citation"),
        ("Факт [S9].", "unknown_citation"),
        ("Факт [S0].", "invalid_citation"),
        ("Факт [Sx].", "invalid_citation"),
    ],
)
def test_invalid_citations_are_not_repaired(text: str, reason: str) -> None:
    backend = RecordedNLI()
    result = NLIFactVerifier(settings(), backend=backend).verify(raw(text))
    assert result.claims[0].reason == reason
    assert backend.calls == [] and result.refused


def test_only_cited_source_is_checked() -> None:
    backend = RecordedNLI()
    verifier = NLIFactVerifier(settings(), backend=backend)
    verifier.verify(
        raw(
            "Факт [S2].",
            contexts={"S1": source("Не использовать"), "S2": source("Использовать", "two")},
        )
    )
    assert backend.calls == [("Использовать", "Факт .")]


def test_duplicate_citations_evaluated_once() -> None:
    backend = RecordedNLI()
    result = NLIFactVerifier(settings(), backend=backend).verify(raw("Факт [S1] [S1]."))
    assert len(backend.calls) == 1 and result.claims[0].source_ids == ("S1",)


def test_conflicting_sources_override_entailment() -> None:
    class ByPremise:
        def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
            return CONTRADICTION if "запрещено" in premise else ENTAILMENT

    result = NLIFactVerifier(settings(), backend=ByPremise()).verify(
        raw(
            "Разрешено [S1] [S2].",
            contexts={"S1": source("разрешено"), "S2": source("запрещено", "two")},
        )
    )
    assert result.claims[0].status == ClaimStatus.CONTRADICTION
    assert len(result.claims[0].evidence) == 2


def test_overlong_nli_pair_is_unverified() -> None:
    result = NLIFactVerifier(settings(), backend=RecordedNLI(None)).verify(raw("Факт [S1]."))
    assert result.refused
    assert result.claims[0].evidence[0].reason == "pair_exceeds_nli_context"


def test_threshold_equality() -> None:
    p = NLIProbabilities(p_entailment=0.75, p_neutral=0.20, p_contradiction=0.05)
    assert (
        NLIFactVerifier(settings(), backend=RecordedNLI(p))
        .verify(raw("Факт [S1]."))
        .claims[0]
        .status
        == ClaimStatus.VERIFIED
    )
    p = NLIProbabilities(p_entailment=0.10, p_neutral=0.50, p_contradiction=0.40)
    assert (
        NLIFactVerifier(settings(), backend=RecordedNLI(p))
        .verify(raw("Факт [S1]."))
        .claims[0]
        .status
        == ClaimStatus.CONTRADICTION
    )


def test_truncated_completion_and_refusal() -> None:
    backend = RecordedNLI()
    verifier = NLIFactVerifier(settings(), backend=backend)
    assert verifier.verify(raw("Факт [S1].", finish="length")).refused
    backend.calls.clear()
    assert verifier.verify(raw(REFUSAL)).claims == ()
    assert verifier.verify(raw("")).refused and backend.calls == []


def test_probabilities_are_validated() -> None:
    with pytest.raises(ValidationError):
        NLIProbabilities(p_entailment=float("nan"), p_neutral=0, p_contradiction=0)
    with pytest.raises(ValidationError, match="sum to one"):
        NLIProbabilities(p_entailment=0.8, p_neutral=0.8, p_contradiction=0.8)


def test_label_mapping_uses_semantics() -> None:
    assert (
        label_indices({"0": "contradiction", "1": "entailment", "2": "neutral"})["entailment"] == 1
    )
    with pytest.raises(VerificationError):
        label_indices({"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"})


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(self, system_prompt: str, prompt: str) -> ChatReply:
        self.calls.append((system_prompt, prompt))
        return ChatReply(
            text="Давление 10 МПа [S1].",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=8,
        )

    def close(self) -> None:
        self.closed = True


def chunk(text: str, identifier: str = "one") -> DocumentChunk:
    return DocumentChunk(identifier, text, "gost", "gost.pdf", ("4.2",), 3, 3, (), 10)


def test_generator_prompt_and_metadata() -> None:
    backend = FakeChat()
    generator = LocalLLMGenerator(settings(), backend=backend)
    result = generator.generate("Какое давление?", [chunk("10 МПа")])
    assert result.contexts["S1"].text == "10 МПа"
    assert result.metadata.completion_tokens == 8
    assert REFUSAL in backend.calls[0][0]
    assert json.loads(backend.calls[0][1])["sources"]["S1"]["page_start"] == 3
    generator.close()
    assert backend.closed
    with pytest.raises(VerificationError, match="closed"):
        generator.generate("вопрос", [])


def test_empty_context_does_not_infer() -> None:
    backend = FakeChat()
    generator = LocalLLMGenerator(settings(), backend=backend)
    assert generator.generate("вопрос", []).text == REFUSAL
    assert not backend.calls
    with pytest.raises(ValueError):
        generator.generate(" ", [chunk("текст")])


def test_context_packing_keeps_original_source_ids() -> None:
    backend = FakeChat()
    generator = LocalLLMGenerator(
        settings(llm_context_window=1024, llm_max_tokens=32), backend=backend
    )
    result = generator.generate("вопрос", [chunk("слово " * 2000), chunk("Короткий текст", "two")])
    assert list(result.contexts) == ["S2"] and result.metadata.omitted_sources == ("S1",)


@pytest.mark.parametrize(
    "error,expected_attempts", [("CUDA out of memory", [-1, 0]), ("invalid GGUF", [-1])]
)
def test_gpu_retry_does_not_mask_corrupt_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str, expected_attempts: list[int]
) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"test-layout")
    attempts = []

    class Constructor:
        def __init__(self, *, n_gpu_layers: int, **kwargs: Any) -> None:
            attempts.append(n_gpu_layers)
            if n_gpu_layers == -1:
                raise RuntimeError(error)

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Constructor))
    if len(expected_attempts) == 1:
        with pytest.raises(VerificationError, match="initialization failed"):
            _LlamaBackend(settings(llm_model_path=path))
    else:
        assert _LlamaBackend(settings(llm_model_path=path)).cpu_retry
    assert attempts == expected_attempts


def test_missing_weights_in_production_raise_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_MODE", "production")
    monkeypatch.setenv("RAG_ALLOW_FALLBACK", "false")
    monkeypatch.setenv("RAG_API_TOKENS", '["01234567890123456789012345678901"]')
    for constructor in (LocalLLMGenerator, NLIFactVerifier):
        with pytest.raises(RuntimeError):
            constructor()


def test_development_fallback_is_explicit() -> None:
    generator = LocalLLMGenerator(settings())
    result = generator.generate("вопрос", [chunk("текст")])
    assert result.text == REFUSAL and result.metadata.fallback_reason
    verifier = NLIFactVerifier(settings())
    assert verifier.fallback_reason
    assert not verifier.verify(raw("Давление составляет 10 МПа [S1].")).is_reliable
