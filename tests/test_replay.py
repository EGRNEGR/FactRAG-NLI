"""Offline replay helpers: dev/test split, ablation ladder, memoized NLI, re-verified drafts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from eval_dataset import load_dataset
from evaluate_suite import (
    ABLATION,
    BASELINE_VERIFIER,
    CaseLog,
    MemoNLI,
    ablation_steps,
    archive_failed,
    case_row,
    dataset_split,
    outcome,
    replay_case,
    summarize_e2e,
)
from generator_verifier import (
    REFUSAL,
    GenerationMetadata,
    NLIFactVerifier,
    NLIProbabilities,
    RawGenerationResult,
    SourceContext,
)
from pipeline import refusal_reason
from settings import RAGSettings

ROOT = Path(__file__).resolve().parents[1]
DATASET = load_dataset(ROOT / "data" / "eval_dataset_v3.json")
SUPPORT = NLIProbabilities(p_entailment=0.95, p_neutral=0.04, p_contradiction=0.01)
NEUTRAL = NLIProbabilities(p_entailment=0.10, p_neutral=0.85, p_contradiction=0.05)


def test_split_is_deterministic_and_halves_every_category() -> None:
    split = dataset_split(DATASET.cases)
    assert split == dataset_split(reversed(DATASET.cases))
    assert set(split) == {c.id for c in DATASET.cases}
    for category in {c.category for c in DATASET.cases}:
        counts = Counter(split[c.id] for c in DATASET.cases if c.category == category)
        assert 0 <= counts["dev"] - counts["test"] <= 1


def test_ablation_ladder_starts_at_the_baseline_and_ends_at_the_current_profile() -> None:
    steps = ablation_steps()
    assert [name for name, _ in steps] == [name for name, _ in ABLATION]
    assert steps[0][1] == BASELINE_VERIFIER and steps[-1][1] == {}
    released = [key for _, keys in ABLATION for key in keys]
    assert sorted(released) == sorted(BASELINE_VERIFIER)
    defaults = RAGSettings.model_validate({"mode": "development", "allow_fallback": True})
    assert all(getattr(defaults, key) != value for key, value in BASELINE_VERIFIER.items())


class Counting:
    def __init__(self) -> None:
        self.single: list[tuple[str, str]] = []
        self.batches: list[list[tuple[str, str]]] = []
        self.last_premise: str | None = None

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        self.single.append((premise, hypothesis))
        self.last_premise = premise.upper()
        return SUPPORT if "yes" in premise else NEUTRAL

    def predict_many(self, pairs: list[tuple[str, str]]) -> list[NLIProbabilities]:
        self.batches.append(list(pairs))
        return [SUPPORT if "yes" in p else NEUTRAL for p, _ in pairs]


def test_memo_scores_each_input_once_and_keeps_batches_whole() -> None:
    backend = Counting()
    memo = MemoNLI(backend)
    assert memo.predict("yes", "h") == SUPPORT and memo.predict("yes", "h") == SUPPORT
    assert backend.single == [("yes", "h")] and memo.last_premise == "YES"
    assert memo.predict_many([("yes", "h"), ("no", "h")]) == [SUPPORT, NEUTRAL]
    assert memo.predict_many([("no", "h"), ("yes", "h")]) == [NEUTRAL, SUPPORT]
    assert backend.batches == [[("yes", "h"), ("no", "h")]]
    memo.predict_many([("yes", "h"), ("new", "h")])
    assert backend.batches[-1] == [("yes", "h"), ("new", "h")]  # rescored as one batch


def logged_response(text: str, finish: str = "stop") -> dict[str, Any]:
    settings = RAGSettings.model_validate({"mode": "development", "allow_fallback": True})
    contexts = {
        "S1": SourceContext(chunk_id="c1", document_id="d", text="yes: source", citation="d, 1")
    }
    verified = NLIFactVerifier(settings, backend=Counting()).verify(
        RawGenerationResult(
            text=text,
            contexts=contexts,
            metadata=GenerationMetadata(backend="test", finish_reason=finish),
        )
    )
    return {
        "answer": verified.cleaned_text,
        "raw_answer": verified.raw_text,
        "refused": verified.refused,
        "refusal_reason": refusal_reason(verified),
        "sources": [{"chunk_id": "c1"}],
        "verification_attempts": [verified.model_dump(mode="json")],
    }


def test_replay_reverifies_the_first_draft_with_the_given_verifier() -> None:
    case = next(c for c in DATASET.cases if c.expected_behavior == "answer")
    response = logged_response("Первый факт. Второй факт [S1].")
    settings = RAGSettings.model_validate({"mode": "development", "allow_fallback": True})
    legacy = settings.model_copy(update={"claim_citation_inheritance": False})
    current = replay_case(case, response, NLIFactVerifier(settings, backend=Counting()))
    assert current["ok"] and not current["refused"]
    assert [c["citation_inherited"] for c in current["claims"]] == [True, False]
    assert current["raw_answer"] == "Первый факт. Второй факт [S1]."
    old = replay_case(case, response, NLIFactVerifier(legacy, backend=Counting()))
    assert [c["reason"] for c in old["claims"]][0] == "missing_citation"
    assert old["vanilla"] == current["vanilla"]  # the draft is the same in every step
    summary = summarize_e2e([current, old], {c.id: c for c in DATASET.cases})
    assert summary["claims"]["total"] == 4 and summary["errors"] == 0
    assert summary["repair"]["attempted"] == 0


def test_failed_cases_are_archived_and_rerun(tmp_path: Path) -> None:
    log = CaseLog(tmp_path / "cases.jsonl")
    log.append({"id": "A", "ok": True})
    log.append({"id": "B", "ok": False, "error": "VerificationError: timed out"})
    assert archive_failed(log, tmp_path / "retried_errors.jsonl") == ["B"]
    assert CaseLog(tmp_path / "cases.jsonl").done == {"A"}
    assert CaseLog(tmp_path / "retried_errors.jsonl").rows[0]["error"].endswith("timed out")
    assert archive_failed(log, tmp_path / "retried_errors.jsonl") == []


def test_vanilla_is_the_first_draft_when_a_repair_attempt_was_published() -> None:
    case = next(c for c in DATASET.cases if c.expected_behavior == "refusal")
    row = case_row(case) | outcome(
        case,
        answer="Смежный факт [S1].",
        raw_answer=REFUSAL,
        final_draft="Смежный факт [S1].",
        attempts=2,
        refused=False,
        refusal_reason=None,
        claims=[],
        sources=[],
        finish_reason="stop",
    )
    assert row["vanilla"]["correct"] and row["factrag"]["hallucination"]
    summary = summarize_e2e([row], {c.id: c for c in DATASET.cases})
    assert summary["repair"] == {
        "attempted": 1,
        "published": 1,
        "factrag_correct": 0,
        "factrag_hallucinations": 1,
    }


def test_replay_keeps_gate_refusals_and_reports_refusal_causes() -> None:
    case = next(c for c in DATASET.cases if c.expected_behavior == "refusal")
    gate: dict[str, Any] = {
        "answer": REFUSAL,
        "raw_answer": None,
        "refused": True,
        "refusal_reason": "relevance_threshold_not_met",
        "sources": [],
        "verification_attempts": [],
    }
    row = replay_case(case, gate, None)
    assert row["refusal_reason"] == "relevance_threshold_not_met" and row["factrag"]["correct"]
    truncated = logged_response("Факт [S1].", finish="length")
    assert truncated["refusal_reason"] == "generation_incomplete"
    assert logged_response(REFUSAL)["refusal_reason"] == "llm_refusal"
    assert logged_response("Факт [S1].")["refusal_reason"] is None


def test_co_verification_stress_counts_only_pairs_with_a_supported_first_claim() -> None:
    from types import SimpleNamespace

    from evaluate_suite import co_verification_stress

    sentence = "Every endpoint must advertise its idle timeout to the peer in parameters."
    probe = SimpleNamespace(
        label="neutral", hypothesis_en="The idle timeout is negotiated.", hypothesis_ru="Таймаут."
    )
    cases = [
        SimpleNamespace(id="N1", nli_probe=probe),
        SimpleNamespace(id="E1", nli_probe=SimpleNamespace(label="entailment")),
    ]
    contexts = {
        "N1": SourceContext(chunk_id="c", document_id="d", text=f"{sentence} Short.", citation="S1")
    }

    class SupportFirst:
        def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
            return SUPPORT if hypothesis.startswith("Every endpoint") else NEUTRAL

    settings = RAGSettings.model_validate({"mode": "development", "allow_fallback": True})
    verifier = NLIFactVerifier(settings, backend=SupportFirst())
    dataset: Any = SimpleNamespace(cases=cases)  # only .cases is read
    report = co_verification_stress(verifier, dataset, contexts)
    assert report["pairs"] == 2 and report["leaked"]["k"] == 1
    assert report["leaked_by_co_verification"] == 1 and report["leaked_ids"] == ["N1:en"]
