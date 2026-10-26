"""Text-level verification helpers: segmentation, inheritance, premises, anchors, meta claims."""

from __future__ import annotations

import pytest

from grounding import (
    anchors_present,
    cited_claims,
    claim_anchors,
    clean_premise,
    is_absence_statement,
    is_verdict_marker,
    premise_windows,
    segment_claims,
    verbatim_code_span,
)
from hybrid_retriever import coverage_weight


@pytest.mark.parametrize(
    "text,v1,v2",
    [
        (
            "Контроли SP 800-53 Rev. 5 разделены на 20 семейств [S1].",
            ["Контроли SP 800-53 Rev.", "5 разделены на 20 семейств [S1]."],
            ["Контроли SP 800-53 Rev. 5 разделены на 20 семейств [S1]."],
        ),
        (
            "Факт [S1]; другой факт [S2].",
            ["Факт [S1];", "другой факт [S2]."],
            ["Факт [S1]; другой факт [S2]."],
        ),
        (
            "Создаётся с LIKE ... INCLUDING IDENTITY [S2].",
            ["Создаётся с LIKE ...", "INCLUDING IDENTITY [S2]."],
            ["Создаётся с LIKE ... INCLUDING IDENTITY [S2]."],
        ),
        (
            "Например, e.g. Fig. 3 и No. 1 [S1]. Второй факт [S1], [S2].",
            ["Например, e.g.", "Fig.", "3 и No.", "1 [S1].", "Второй факт [S1], [S2]."],
            ["Например, e.g. Fig. 3 и No. 1 [S1].", "Второй факт [S1], [S2]."],
        ),
    ],
)
def test_segmentation_versions(text: str, v1: list[str], v2: list[str]) -> None:
    assert segment_claims(text) == v1  # default stays the original behaviour
    assert segment_claims(text, "v2") == v2


def test_sentence_end_after_abbreviation_without_number_still_splits() -> None:
    assert segment_claims("Ответ: no. Следующий факт [S1].", "v2") == [
        "no.",
        "Следующий факт [S1].",
    ]


def test_citation_inheritance_order() -> None:
    claims = cited_claims("Факт 1. Факт 2 [S1].\nФакт 3.\nФакт 4 [S2]. Факт 5.")
    assert [(c.text, c.source_ids, c.inherited) for c in claims] == [
        ("Факт 1.", ("S1",), True),  # next citation in the same paragraph
        ("Факт 2 [S1].", ("S1",), False),
        ("Факт 3.", ("S1",), True),  # nearest preceding citation in the answer
        ("Факт 4 [S2].", ("S2",), False),
        ("Факт 5.", ("S2",), True),  # preceding citation in the same paragraph
    ]


def test_no_inheritance_without_any_citation_or_for_malformed_markers() -> None:
    assert [c.source_ids for c in cited_claims("Факт. Другой факт.")] == [(), ()]
    assert cited_claims("Факт [S0]. Другой [S1].")[0].source_ids == ()
    assert [c.inherited for c in cited_claims("Факт. Другой [S1].", inherit=False)] == [
        False,
        False,
    ]


def test_premise_cleanup_and_windows() -> None:
    raw = (
        'Документ: abc\nРаздел: 1 Intro\n<code class="x"><span>PYTHONGIL=0</span></code> &amp; on.'
    )
    assert clean_premise(raw) == "PYTHONGIL=0 & on."
    assert clean_premise(raw, strip_markup=False).startswith("<code")
    assert premise_windows("A one. B two. C three.") == [
        "A one.",
        "B two.",
        "C three.",
        "A one. B two.",
        "B two. C three.",
    ]
    assert premise_windows("Single sentence.") == []


def test_anchors_ignore_document_designators() -> None:
    text = "SP 800-63B-4 требует 30 дней для AAL1; max_ack_delay 16 383 мс, PYTHONGIL=0 [S1]."
    assert claim_anchors(text) == frozenset({"30", "aal1", "max_ack_delay", "16383", "pythongil=0"})
    assert claim_anchors("RFC 9110 определяет методы GET и HEAD как безопасные.") == frozenset()
    assert claim_anchors("traceId хранит 0x03 и 0b0111 для UUIDv7, AC-2k") == frozenset(
        {"traceid", "0x03", "0b0111", "uuidv7", "ac-2k"}
    )


def test_anchor_presence_respects_number_boundaries() -> None:
    assert anchors_present(frozenset({"20"}), "Values above 20 are invalid.")
    assert not anchors_present(frozenset({"20"}), "Published in 2020.")
    assert anchors_present(frozenset({"16383"}), "up to 16 383 bytes")
    assert not anchors_present(frozenset({"application/json+problem"}), "application/problem+json")


def test_verbatim_code_span() -> None:
    premise = "Content-Range = range-unit SP ( range-resp / unsatisfied-range )"
    claim = "Content-Range = range-unit SP (range-resp / unsatisfied-range)."
    assert verbatim_code_span(claim, premise)
    assert not verbatim_code_span(claim, premise, 0)
    assert not verbatim_code_span(
        "The server must send the content.", "The server must send the content."
    )
    assert not verbatim_code_span("Allow = [ method ]", "Allow = #method")


@pytest.mark.parametrize(
    "text,absence",
    [
        ("RFC 9562 не содержит информации о методе randomUUID().", True),
        ("В документации отсутствуют прямые указания на то, как это работает.", True),
        ("The document does not contain any information about X.", True),
        ("Null значения не считаются равными, если не указано NULLS NOT DISTINCT.", False),
        ("Помимо 100-continue, RFC 9110 не определяет других ожиданий.", False),
        ("RFC 9110 does not require clients to retry.", False),
    ],
)
def test_absence_statements(text: str, absence: bool) -> None:
    assert is_absence_statement(text) is absence


def test_verdict_markers() -> None:
    assert is_verdict_marker("Неверно.") and is_verdict_marker("No.")
    assert not is_verdict_marker("Нет, RFC 9110 требует отправлять заголовок Date.")


def test_coverage_weight() -> None:
    vocabulary = {"max_idle_timeout", "quic", "timeout"}
    assert coverage_weight(["max_idle_timeout", "тайм", "аут"], vocabulary, 1.0) == pytest.approx(
        1 / 3
    )
    assert coverage_weight(["таблица"], vocabulary, 2.0) == 0.0
    assert coverage_weight([], vocabulary, 1.5) == 1.5
