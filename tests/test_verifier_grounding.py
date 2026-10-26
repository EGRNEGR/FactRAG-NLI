"""Verifier behaviour of the evaluated profile: windows, anchors, aggregation, meta claims."""

from __future__ import annotations

from typing import Any

from generator_verifier import (
    REFUSAL,
    ClaimStatus,
    GenerationMetadata,
    NLIFactVerifier,
    NLIProbabilities,
    RawGenerationResult,
    SourceContext,
)
from settings import RAGSettings

ENTAILMENT = NLIProbabilities(p_entailment=0.95, p_neutral=0.04, p_contradiction=0.01)
MODERATE = NLIProbabilities(p_entailment=0.55, p_neutral=0.35, p_contradiction=0.10)
NEUTRAL = NLIProbabilities(p_entailment=0.10, p_neutral=0.85, p_contradiction=0.05)
NOISE = NLIProbabilities(p_entailment=0.40, p_neutral=0.30, p_contradiction=0.30)
CONTRADICTION = NLIProbabilities(p_entailment=0.02, p_neutral=0.03, p_contradiction=0.95)

LEGACY = {
    "claim_segmentation": "v1",
    "claim_citation_inheritance": False,
    "nli_premise_cleanup": False,
    "nli_window_scoring": False,
    "nli_anchor_guard": False,
    "nli_anchor_windows": False,
    "nli_code_span_min_chars": 0,
    "nli_aggregation": "strict",
    "abstention_guard": False,
}


def settings(**overrides: Any) -> RAGSettings:
    return RAGSettings.model_validate({"mode": "development", "allow_fallback": True, **overrides})


def source(text: str, identifier: str = "one") -> SourceContext:
    return SourceContext(chunk_id=identifier, document_id="doc", text=text, citation="doc, 1")


def raw(text: str, contexts: dict[str, SourceContext]) -> RawGenerationResult:
    return RawGenerationResult(
        text=text,
        contexts=contexts,
        metadata=GenerationMetadata(backend="test", finish_reason="stop"),
    )


class Table:
    """Probabilities chosen by premise text; anything unlisted is `default`."""

    def __init__(self, table: dict[str, NLIProbabilities], default: NLIProbabilities) -> None:
        self.table, self.default = table, default
        self.calls: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        self.calls.append((premise, hypothesis))
        return self.table.get(premise, self.default)


CHUNK = (
    "A client MUST expand the payload to at least 1200 bytes. "
    "A server MAY exceed 1200 bytes. Datagrams below that size are discarded."
)


def test_window_overrides_false_whole_chunk_contradiction() -> None:
    backend = Table(
        {"A client MUST expand the payload to at least 1200 bytes.": ENTAILMENT}, CONTRADICTION
    )
    result = NLIFactVerifier(settings(), backend=backend).verify(
        raw("Клиент дополняет полезную нагрузку минимум до 1200 байт [S1].", {"S1": source(CHUNK)})
    )
    claim = result.claims[0]
    assert claim.status == ClaimStatus.VERIFIED and not result.refused
    assert claim.evidence[0].support == "window"
    assert (
        claim.evidence[0].premise_text == "A client MUST expand the payload to at least 1200 bytes."
    )
    legacy = NLIFactVerifier(settings(**LEGACY), backend=backend).verify(
        raw("Клиент дополняет полезную нагрузку минимум до 1200 байт [S1].", {"S1": source(CHUNK)})
    )
    assert legacy.claims[0].status == ClaimStatus.CONTRADICTION and legacy.refused


def test_anchor_guard_blocks_number_absent_from_sources() -> None:
    backend = Table({}, ENTAILMENT)
    result = NLIFactVerifier(settings(), backend=backend).verify(
        raw("Клиент дополняет нагрузку до 1500 байт [S1].", {"S1": source(CHUNK)})
    )
    assert result.claims[0].status == ClaimStatus.UNVERIFIED
    assert result.claims[0].reason == "anchor_not_in_source" and result.refused


def test_anchor_window_needs_every_anchor_in_that_window() -> None:
    window = "A server MAY exceed 1200 bytes."
    backend = Table({window: MODERATE}, NEUTRAL)
    supported = NLIFactVerifier(settings(), backend=backend).verify(
        raw("Сервер может превышать 1200 байт [S1].", {"S1": source(CHUNK)})
    )
    assert supported.claims[0].evidence[0].support == "anchor_window"
    unanchored = NLIFactVerifier(settings(), backend=backend).verify(
        raw("Сервер может превышать размер [S1].", {"S1": source(CHUNK)})
    )
    assert unanchored.claims[0].status == ClaimStatus.UNVERIFIED


def test_verbatim_code_span_supports_grammar_rule() -> None:
    rule = 'If-Match = "*" / #entity-tag'
    backend = Table({}, NOISE)
    result = NLIFactVerifier(settings(), backend=backend).verify(
        raw(f"ABNF поля: {rule} [S1].", {"S1": source(f"The field is defined as {rule} below.")})
    )
    assert result.claims[0].evidence[0].support == "code_span"
    assert result.claims[0].status == ClaimStatus.VERIFIED


def test_support_first_ignores_noise_but_not_a_strong_contradiction() -> None:
    contexts = {"S1": source("разрешено"), "S2": source("смежный раздел", "two")}
    noisy = Table({"разрешено": ENTAILMENT, "смежный раздел": NOISE}, NEUTRAL)
    assert (
        NLIFactVerifier(settings(), backend=noisy)
        .verify(raw("Разрешено [S1][S2].", contexts))
        .claims[0]
        .status
        == ClaimStatus.VERIFIED
    )
    assert (
        NLIFactVerifier(settings(**LEGACY), backend=noisy)
        .verify(raw("Разрешено [S1][S2].", contexts))
        .claims[0]
        .status
        == ClaimStatus.CONTRADICTION
    )
    vetoed = Table({"разрешено": ENTAILMENT, "смежный раздел": CONTRADICTION}, NEUTRAL)
    assert (
        NLIFactVerifier(settings(), backend=vetoed)
        .verify(raw("Разрешено [S1][S2].", contexts))
        .claims[0]
        .status
        == ClaimStatus.CONTRADICTION
    )


def test_inherited_citation_is_verified_and_published_with_its_marker() -> None:
    backend = Table({}, ENTAILMENT)
    result = NLIFactVerifier(settings(), backend=backend).verify(
        raw("Первый факт. Второй факт [S1].", {"S1": source("Первый факт. Второй факт.")})
    )
    assert [c.citation_inherited for c in result.claims] == [True, False]
    assert result.cleaned_text == "Первый факт. [S1]\nВторой факт [S1]."
    legacy = NLIFactVerifier(settings(**LEGACY), backend=backend).verify(
        raw("Первый факт. Второй факт [S1].", {"S1": source("Первый факт. Второй факт.")})
    )
    assert legacy.claims[0].reason == "missing_citation"


def test_absence_statement_is_kept_and_not_counted_as_blocked() -> None:
    backend = Table({}, ENTAILMENT)
    text = "RFC 9562 не содержит информации о методе Java. Однако UUIDv4 основан на случайных числах [S1]."
    result = NLIFactVerifier(settings(), backend=backend).verify(
        raw(text, {"S1": source("UUIDv4 uses random numbers.")})
    )
    assert [c.kind for c in result.claims] == ["absence", "fact"]
    assert result.hallucination_rate == 0 and not result.refused
    assert result.cleaned_text.startswith("RFC 9562 не содержит информации")


def test_absence_only_answer_becomes_the_canonical_refusal() -> None:
    result = NLIFactVerifier(settings(), backend=Table({}, ENTAILMENT)).verify(
        raw("Документ не содержит сведений о таймауте.", {"S1": source("Other text.")})
    )
    assert result.refused and result.cleaned_text == REFUSAL


def test_premise_correction_is_not_published_without_its_justification() -> None:
    contexts = {"S1": source("A client SHOULD NOT retry non-idempotent requests.")}
    text = "RFC 9110 не требует повторять запросы [S1]. Клиент SHOULD NOT повторять такие запросы [S1]."
    backend = Table({}, NEUTRAL)
    backend.table["A client SHOULD NOT retry non-idempotent requests."] = NEUTRAL

    class FirstOnly:
        def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
            return ENTAILMENT if "не требует" in hypothesis else NEUTRAL

    result = NLIFactVerifier(settings(), backend=FirstOnly()).verify(raw(text, contexts))
    assert result.claims[0].status == ClaimStatus.VERIFIED
    assert result.refused and result.cleaned_text == REFUSAL
    legacy = NLIFactVerifier(settings(**LEGACY), backend=FirstOnly()).verify(raw(text, contexts))
    assert not legacy.refused


def test_verdict_marker_travels_with_the_next_claim() -> None:
    result = NLIFactVerifier(settings(), backend=Table({}, ENTAILMENT)).verify(
        raw(
            "Неверно. Версия 1 использует номер 0x00000001 [S1].",
            {"S1": source("Version 1 is 0x00000001.")},
        )
    )
    assert [c.kind for c in result.claims] == ["verdict", "fact"]
    assert result.cleaned_text.startswith("Неверно.")


def test_windows_are_scored_in_one_batch_when_supported() -> None:
    class Batched(Table):
        def __init__(self) -> None:
            super().__init__({}, ENTAILMENT)
            self.batches: list[int] = []

        def predict_many(self, pairs: list[tuple[str, str]]) -> list[NLIProbabilities]:
            self.batches.append(len(pairs))
            return [self.predict(p, h) for p, h in pairs]

    backend = Batched()
    NLIFactVerifier(settings(), backend=backend).verify(
        raw("Клиент дополняет до 1200 байт [S1].", {"S1": source(CHUNK)})
    )
    assert backend.batches == [6]  # whole chunk + 3 single + 2 paired sentence windows
