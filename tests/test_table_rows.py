"""optimization_6: NLI windows per table row and anchors over bit cells (both off by default)."""

from __future__ import annotations

from typing import Any

from generator_verifier import (
    ClaimStatus,
    GenerationMetadata,
    NLIFactVerifier,
    NLIProbabilities,
    RawGenerationResult,
    SourceContext,
)
from grounding import markdown_tables, table_bit_runs, table_row_windows
from settings import RAGSettings

ENTAILMENT = NLIProbabilities(p_entailment=0.95, p_neutral=0.04, p_contradiction=0.01)
NEUTRAL = NLIProbabilities(p_entailment=0.10, p_neutral=0.85, p_contradiction=0.05)

IANA = (
    "Документ: rfc9562\nРаздел: 7 IANA Considerations\n"
    "Table 4: IANA UUID Subtypes\n"
    "| Name | ID | Subtype |\n"
    "| --- | --- | --- |\n"
    "| Random | 4 | version |\n"
    "| Unix Time-based | 7 | version |\n"
    "|  |  |  |"
)
ROW_7 = "Table 4: IANA UUID Subtypes: Name: Unix Time-based; ID: 7; Subtype: version."

VERSIONS = (
    "Table 2: UUID Variant 10xx Versions\n"
    "| MSB0 | MSB1 | MSB2 | MSB3 | Version |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| 0 | 1 | 1 | 1 | 7 |\n"
    "| 1 | 0 | 0 | 0 | 8 |"
)


def settings(**overrides: Any) -> RAGSettings:
    return RAGSettings.model_validate({"mode": "development", "allow_fallback": True, **overrides})


def raw(text: str, chunk: str) -> RawGenerationResult:
    return RawGenerationResult(
        text=text,
        contexts={"S1": SourceContext(chunk_id="c", document_id="d", text=chunk, citation="d")},
        metadata=GenerationMetadata(backend="test", finish_reason="stop"),
    )


class ByPremise:
    def __init__(self, table: dict[str, NLIProbabilities]) -> None:
        self.table = table
        self.premises: list[str] = []

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        self.premises.append(premise)
        return self.table.get(premise, NEUTRAL)


class Entails:
    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        return ENTAILMENT


def test_markdown_tables_read_caption_header_and_rows() -> None:
    (table,) = markdown_tables(IANA)
    assert table.caption == "Table 4: IANA UUID Subtypes"
    assert table.header == ("Name", "ID", "Subtype")
    assert table.rows[1] == ("Unix Time-based", "7", "version")


def test_row_windows_bind_values_to_headers_and_skip_empty_rows() -> None:
    assert table_row_windows(IANA) == [
        "Table 4: IANA UUID Subtypes: Name: Random; ID: 4; Subtype: version.",
        ROW_7,
    ]


def test_repeated_header_splits_a_row_into_groups() -> None:
    text = "| ID | FAMILY | ID | FAMILY |\n| --- | --- | --- | --- |\n| AC | Access | PE | Physical |"
    assert table_row_windows(text) == ["ID: AC; FAMILY: Access.", "ID: PE; FAMILY: Physical."]


def test_no_row_windows_for_one_column_or_headerless_tables() -> None:
    note = "| *Note:* a user agent MAY change the |\n| --- |\n| request method. |"
    synopsis = "| CREATE [ GLOBAL | LOCAL ] TABLE ( |  |\n| --- | --- | --- |\n| a | b | c |"
    prose = "No table here. Just text."
    assert table_row_windows(note) == table_row_windows(synopsis) == table_row_windows(prose) == []


def test_bit_runs_are_maximal_runs_of_one_bit_cells() -> None:
    assert table_bit_runs(VERSIONS) == ["0111", "1000"]
    # A pipe-less chunk or a table of multi-digit numbers has no runs.
    assert table_bit_runs("| A | B |\n| --- | --- |\n| 10 | 11 |") == []


def test_row_window_supports_a_claim_only_when_enabled() -> None:
    claim = "Описание версии 7 в таблице IANA - Unix Time-based [S1]."
    off = NLIFactVerifier(settings(), backend=ByPremise({ROW_7: ENTAILMENT})).verify(
        raw(claim, IANA)
    )
    assert off.claims[0].status == ClaimStatus.UNVERIFIED
    backend = ByPremise({ROW_7: ENTAILMENT})
    on = NLIFactVerifier(settings(nli_table_row_windows=True), backend=backend).verify(
        raw(claim, IANA)
    )
    assert on.claims[0].status == ClaimStatus.VERIFIED and not on.refused
    assert on.claims[0].evidence[0].premise_text == ROW_7
    assert ROW_7 in backend.premises


def test_plain_text_premises_are_unchanged_by_the_flag() -> None:
    chunk = "A client MUST expand the payload to at least 1200 bytes. A server MAY exceed it."
    claim = "Клиент дополняет нагрузку до 1200 байт [S1]."
    off, on = ByPremise({}), ByPremise({})
    NLIFactVerifier(settings(), backend=off).verify(raw(claim, chunk))
    NLIFactVerifier(settings(nli_table_row_windows=True), backend=on).verify(raw(claim, chunk))
    assert on.premises == off.premises


def test_bit_cells_satisfy_the_anchor_guard_only_when_enabled() -> None:
    claim = "Версии 8 соответствует битовый шаблон 1000 [S1]."
    backend = Entails()
    off =NLIFactVerifier(settings(), backend=backend).verify(raw(claim, VERSIONS))
    assert off.claims[0].reason == "anchor_not_in_source"
    on = NLIFactVerifier(settings(nli_anchor_bit_cells=True), backend=backend).verify(
        raw(claim, VERSIONS)
    )
    assert on.claims[0].status == ClaimStatus.VERIFIED
    # A partial run is not an anchor match.
    partial = NLIFactVerifier(settings(nli_anchor_bit_cells=True), backend=backend).verify(
        raw("Версии 8 соответствует шаблон 100 [S1].", VERSIONS)
    )
    assert partial.claims[0].reason == "anchor_not_in_source"
