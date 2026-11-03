"""optimization_4 variants: anchor scope "document" (E1), one-column tables (E2), --ids."""

from __future__ import annotations

import pytest

from evaluate_suite import select_cases
from generator_verifier import ClaimStatus, NLIFactVerifier, SourceContext
from grounding import claim_anchors, clean_premise, is_identifier_anchor, unwrap_single_column
from tests.test_verifier_grounding import ENTAILMENT, Table, raw, settings

NOTE = (
    "Документ: rfc\nРаздел: 15.4.3 302 Found\n"
    "| *Note:* For historical reasons, a user agent MAY change the |\n"
    "| --- |\n"
    "| request method from POST to GET for the subsequent request. |\n"
    "Next paragraph."
)
WIDE = "|CONTROL NUMBER|CONTROL NAME|\n|---|---|\n|AC-7(4)|USE OF ALTERNATE AUTHENTICATION FACTOR|"


def test_one_column_table_becomes_running_text() -> None:
    assert unwrap_single_column(NOTE.split("\n", 2)[2]) == (
        "Note: For historical reasons, a user agent MAY change the request method from POST "
        "to GET for the subsequent request.\nNext paragraph."
    )
    assert unwrap_single_column(WIDE) == WIDE  # a real table keeps its structure
    assert clean_premise(NOTE).startswith("| *Note:*")  # off by default
    assert clean_premise(NOTE, unwrap_tables=True).startswith("Note: For historical")


def test_enhancement_designator_is_one_identifier() -> None:
    text = "Усиление AC-7(4) разрешает альтернативные факторы [S1]."
    assert claim_anchors(text) == frozenset({"ac-7", "4"})
    assert claim_anchors(text, enhancements=True) == frozenset({"ac-7(4)"})
    assert is_identifier_anchor("ac-7(4)") and is_identifier_anchor("uuidv7")
    for literal in ("10", "0x03", "0b0111", "pythongil=1", "--disable-gil", "max_ack_delay"):
        assert not is_identifier_anchor(literal)


def contexts() -> dict[str, SourceContext]:
    body = SourceContext(
        chunk_id="body",
        document_id="sp800-53",
        text="(a) Allow the use of authentication factors that are different from the primary "
        "authentication factors after the number of invalid logon attempts have been exceeded.",
        citation="sp800-53",
        section_path=("3 [Assignment: organization-defined attributes (as required)];",),
    )
    label = SourceContext(chunk_id="label", document_id="sp800-53", text=WIDE, citation="t")
    other = SourceContext(
        chunk_id="other", document_id="sp800-63b", text="AC-7(4) is cited here.", citation="o"
    )
    return {"S1": body, "S2": label, "S3": other}


@pytest.mark.parametrize("scope, verified", [("cited", False), ("document", True)])
def test_identifier_from_a_same_document_chunk(scope: str, verified: bool) -> None:
    claim = "Усиление AC-7(4) разрешает факторы, отличные от основных [S1]."
    result = NLIFactVerifier(
        settings(nli_anchor_scope=scope), backend=Table({}, ENTAILMENT)
    ).verify(raw(claim, contexts()))
    assert (result.claims[0].status == ClaimStatus.VERIFIED) is verified
    if not verified:
        assert result.claims[0].reason == "anchor_not_in_source"


def test_document_scope_never_takes_numbers_or_other_documents() -> None:
    c = contexts()
    verifier = NLIFactVerifier(settings(nli_anchor_scope="document"), backend=Table({}, ENTAILMENT))
    # A number missing from the cited chunk stays missing, even if another chunk has it.
    c["S2"] = c["S2"].model_copy(update={"text": WIDE + " Limit of 10 attempts."})
    number = verifier.verify(raw("После 10 неудачных попыток разрешены другие факторы [S1].", c))
    assert number.claims[0].reason == "anchor_not_in_source"
    # The identifier only in a chunk of another document does not count.
    del c["S2"]
    alien = verifier.verify(raw("Усиление AC-7(4) разрешает другие факторы [S1].", c))
    assert alien.claims[0].reason == "anchor_not_in_source"


def test_document_scope_keeps_whatever_passed_in_the_cited_source() -> None:
    # "AC-2(10)" passes in the cited source where "AC-2" and "(10)" stand apart; the enhancement
    # form of the document scope must not make it stricter (V3-LEX-016 in the 1310Z replay).
    cited = SourceContext(
        chunk_id="c", document_id="d", text="AC-2 ACCOUNT MANAGEMENT ... (10) SHARED AND GROUP "
        "ACCOUNT CREDENTIAL CHANGE [Withdrawn: Incorporated into AC-2k.]", citation="c"
    )
    claim = "Пометка Withdrawn у улучшения AC-2(10) означает включение в AC-2k [S1]."
    for scope in ("cited", "document"):
        result = NLIFactVerifier(
            settings(nli_anchor_scope=scope), backend=Table({}, ENTAILMENT)
        ).verify(raw(claim, {"S1": cited}))
        assert result.claims[0].status == ClaimStatus.VERIFIED


def test_select_cases_keeps_documents_and_rejects_foreign_lists() -> None:
    from eval_dataset import load_dataset
    from evaluate_suite import DATASET

    dataset = load_dataset(DATASET)
    ids = [dataset.cases[3].id, dataset.cases[0].id]
    subset = select_cases(dataset, ",".join(ids) + ",V3-XQFEN-999")
    assert [c.id for c in subset.cases] == [dataset.cases[0].id, dataset.cases[3].id]
    assert subset.documents == dataset.documents
    assert select_cases(dataset, None) is dataset
    with pytest.raises(SystemExit):
        select_cases(dataset, "V3-XQFEN-999")
