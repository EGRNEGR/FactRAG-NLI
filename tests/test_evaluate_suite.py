"""Metric, scoring and schema checks for the v3 evaluation suite (no model weights)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from document_parser import DocumentChunk
from eval_dataset import DraftCase, contains_fact, load_dataset
from evaluate_suite import (
    classification_report,
    decision_metrics,
    evidence_metrics,
    latency_stats,
    mcnemar_exact,
    rerank_order,
    score_answer,
    single_source_status,
    summarize_variant,
    threshold_sweep,
    wilson,
)
from generator_verifier import REFUSAL
from hybrid_retriever import HashingEncoder, HybridRetriever, LexicalReranker
from settings import RAGSettings

ROOT = Path(__file__).resolve().parents[1]


def draft(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "V3-SH-999",
        "category": "single_hop",
        "subtype": "definition",
        "language": "en",
        "document": "doc.txt",
        "question": "What does the spec define?",
        "ground_truth_answer": "It defines X.",
        "expected_behavior": "answer",
        "quotes": ["defines X"],
        "nli_probe": {
            "premise_quote": "The spec defines X here.",
            "hypothesis_en": "The spec defines X.",
            "hypothesis_ru": "Спецификация определяет X.",
            "label": "entailment",
        },
        "answer_must_contain": ["X"],
    }
    row.update(overrides)
    return row


def case(**overrides: Any) -> Any:
    values = {"expected_behavior": "answer", "answer_must_contain": ["32"]}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_evidence_metrics_unit_level_recall_and_bounded_ndcg() -> None:
    m = evidence_metrics(["x", "a2", "b1"], [["a1", "a2"], ["b1"]])
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["hit@3"] == 1
    assert m["mrr@10"] == pytest.approx(1 / 2)
    shared = evidence_metrics(["both"], [["both"], ["both"]])
    assert shared["recall@1"] == 1
    assert 0 < shared["ndcg@1"] <= 1
    assert evidence_metrics(["z"], [["a"]])["mrr@10"] == 0
    with pytest.raises(ValueError):
        evidence_metrics(["a"], [])


def test_rerank_order_uses_production_mix_and_skips_headings() -> None:
    settings = SimpleNamespace(dense_weight=1.0, sparse_weight=1.0, rrf_k=60, rerank_rrf_alpha=0.8)
    rows = [
        {"chunk_id": "low", "rrf_score": 0.001, "rerank_score": 0.9},
        {"chunk_id": "high", "rrf_score": 2 / 61, "rerank_score": 0.1},
        {"chunk_id": "head", "rrf_score": 2 / 61, "rerank_score": 1.0, "heading_only": True},
    ]
    assert rerank_order(rows, settings) == ["high", "low"]


def test_classification_report_confusion_and_macro_f1() -> None:
    gold = ["entailment", "entailment", "neutral", "contradiction"]
    pred = ["entailment", "neutral", "neutral", "entailment"]
    report = classification_report(gold, pred)
    assert report["confusion_matrix"]["entailment"]["neutral"] == 1
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["per_class"]["contradiction"]["recall"] == 0
    assert report["per_class"]["neutral"]["precision"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        classification_report(gold, pred[:2])


def test_decision_metrics_catch_rate_and_false_rejection() -> None:
    rows = [
        {"label": "entailment", "status": "verified"},
        {"label": "entailment", "status": "unverified"},
        {"label": "contradiction", "status": "contradiction"},
        {"label": "neutral", "status": "verified"},
    ]
    m = decision_metrics(rows)
    assert m["false_rejection_rate"] == pytest.approx(0.5)
    assert m["hallucination_catch_rate"] == pytest.approx(0.5)
    assert m["contradiction_catch_rate"] == 1 and m["neutral_catch_rate"] == 0
    assert m["status_agreement"] == pytest.approx(0.5)


def test_single_source_rule_matches_production_order() -> None:
    kw = {"entail": 0.4, "contra": 0.2, "strong": 0.7}
    assert single_source_status(0.9, 0.25, **kw) == "contradiction"
    assert single_source_status(0.45, 0.1, **kw) == "verified"
    assert single_source_status(0.35, 0.1, **kw) == "unverified"
    sweep = threshold_sweep(
        [
            {"label": "entailment", "p_entailment": 0.5, "p_contradiction": 0.0},
            {"label": "neutral", "p_entailment": 0.45, "p_contradiction": 0.0},
        ],
        [0.4, 0.6],
        [0.2],
        None,
    )
    assert [p["false_rejection_rate"] for p in sweep] == [0.0, 1.0]
    assert [p["hallucination_catch_rate"] for p in sweep] == [0.0, 1.0]


def test_latency_and_statistics() -> None:
    stats = latency_stats([1, 2, 3, 4, 100])
    assert stats["p50"] == 3 and stats["max"] == 100 and stats["n"] == 5
    assert latency_stats([])["p95"] is None
    low, high = wilson(45, 55) or [0.0, 0.0]
    assert low == pytest.approx(0.6967, abs=1e-3) and high == pytest.approx(0.8981, abs=1e-3)
    assert mcnemar_exact(4, 2) == pytest.approx(0.6875)
    assert mcnemar_exact(0, 0) == 1.0


def test_score_answer_by_expected_behavior() -> None:
    positive = case()
    assert score_answer(positive, "До 32 колонок [S1].")["correct"]
    assert score_answer(positive, REFUSAL)["refused"]
    assert not score_answer(positive, REFUSAL)["facts_ok"]
    negative = case(expected_behavior="refusal", answer_must_contain=[])
    assert score_answer(negative, REFUSAL)["hallucination"] is False
    assert score_answer(negative, "RFC 9110 не определяет такое поле.")["correct"]
    assert score_answer(negative, "Значение равно 5 секундам [S1].")["hallucination"] is True
    adversarial = case(expected_behavior="refusal_or_correction")
    assert score_answer(adversarial, "Нет, до 32 колонок [S1].")["correct"]
    assert score_answer(adversarial, "Да, до 64 колонок [S1].")["hallucination"] is True


def test_russian_negation_trap_requires_regex_fact() -> None:
    fact = "re:even|(?<!не)четн"
    assert contains_fact("Клиентские потоки имеют чётные идентификаторы.", fact)
    assert not contains_fact("Клиентские потоки имеют нечётные идентификаторы.", fact)
    assert contains_fact("Значение: application/problem+json", "application/problem+json")


def test_summarize_variant_counts_only_defined_hallucinations() -> None:
    rows = [
        {
            "ok": True,
            "category": "unanswerable",
            "language": "ru",
            "expected_behavior": "refusal",
            "v": {"correct": False, "hallucination": True, "refused": False, "facts_ok": False},
        },
        {
            "ok": True,
            "category": "single_hop",
            "language": "en",
            "expected_behavior": "answer",
            "v": {"correct": True, "hallucination": None, "refused": False, "facts_ok": True},
        },
        {"ok": False, "category": "single_hop", "language": "en", "expected_behavior": "answer"},
    ]
    summary = summarize_variant(rows, "v")
    assert summary["hallucination"]["n"] == 1 and summary["hallucination"]["k"] == 1
    assert summary["correct"]["n"] == 2
    assert summary["false_refusal_on_answerable"]["k"] == 0


def test_draft_schema_rejects_inconsistent_labels() -> None:
    DraftCase.model_validate(draft())
    with pytest.raises(ValidationError):
        DraftCase.model_validate(draft(nli_probe={**draft()["nli_probe"], "label": "neutral"}))
    with pytest.raises(ValidationError):
        DraftCase.model_validate(
            draft(id="V3-UNA-999", category="unanswerable", expected_behavior="refusal", quotes=[])
        )
    with pytest.raises(ValidationError):
        DraftCase.model_validate(
            draft(
                category="adversarial",
                expected_behavior="refusal_or_correction",
                nli_probe={**draft()["nli_probe"], "label": "contradiction"},
            )
        )
    with pytest.raises(ValidationError):
        DraftCase.model_validate(draft(category="multi_hop"))


def test_committed_dataset_is_stratified_and_consistent() -> None:
    dataset = load_dataset(ROOT / "data" / "eval_dataset_v3.json")
    counts = Counter(c.category for c in dataset.cases)
    assert sum(counts.values()) >= 165  # at least 3x the 55-case v2 set
    assert set(counts) == {
        "single_hop",
        "multi_hop",
        "tabular_code",
        "adversarial",
        "unanswerable",
        "lexical_trap",
    }
    mapping = {
        "answer": "entailment",
        "refusal": "neutral",
        "refusal_or_correction": "contradiction",
    }
    for item in dataset.cases:
        assert item.expected_nli_verdict == mapping[item.expected_behavior]
        assert bool(item.evidence) == item.answerable
        if item.category == "multi_hop":
            sets = [set(e.chunk_ids) for e in item.evidence]
            assert any(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1 :])
    stats = json.loads((ROOT / "data" / "eval_dataset_v3.json").read_text(encoding="utf-8"))[
        "stats"
    ]
    assert stats["cases"] == len(dataset.cases)


def test_retriever_exposes_single_signal_rankings(tmp_path: Path) -> None:
    settings = RAGSettings.model_validate(
        {
            "mode": "development",
            "allow_fallback": True,
            "qdrant_path": tmp_path / "q",
            "rerank_score_threshold": 0.0,
        }
    )
    base = DocumentChunk("a", "alpha bravo", "doc", "doc.txt", ("1",), None, None, (), 4, {})
    chunks = [
        base,
        replace(base, chunk_id="b", text="charlie delta"),
        replace(base, chunk_id="c", text="alpha charlie"),
    ]
    with HybridRetriever(HashingEncoder(64), LexicalReranker(), settings=settings) as search:
        assert search.candidate_lists() == {"dense": [], "sparse": []}
        search.add(chunks)
        _, _, hybrid, _ = search.search_trace("alpha")
        lists = search.candidate_lists()
    assert set(lists["sparse"]) == {"a", "c"}
    assert set(lists["dense"]) <= {"a", "b", "c"}
    assert set(hybrid) <= set(lists["dense"]) | set(lists["sparse"])
