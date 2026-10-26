"""Metric mathematics, grounded labels, immutable corpus and end-to-end reporting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evaluate_quality import GoldenDataset, cosine, evaluate, ranking_metrics, summarize
from generator_verifier import ChatReply, HeuristicNLI, LocalLLMGenerator, NLIFactVerifier
from hybrid_retriever import HashingEncoder, HybridRetriever, LexicalReranker
from pipeline import RAGPipeline
from scripts.download_quality_corpus import TextExtractor
from scripts.prepare_golden_dataset import prepare
from settings import RAGSettings
from baseline_rag import main as cli_main


def test_graded_rank_metrics() -> None:
    metrics = ranking_metrics(["x", "a", "b"], {"a": 3, "b": 1}, 3)
    assert metrics["hit_at_k"] == 1 and metrics["mrr_at_k"] == 0.5
    expected = (7 / math.log2(3) + 1 / math.log2(4)) / (7 + 1 / math.log2(3))
    assert metrics["ndcg_at_k"] == pytest.approx(expected)
    assert ranking_metrics(["x"], {"a": 1}, 1)["hit_at_k"] == 0
    assert ranking_metrics(["a", "a"], {"a": 1}, 3)["ndcg_at_k"] == 1


def test_cosine_and_invalid_embeddings() -> None:
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([1, 0], [-1, 0]) == -1
    for first, second in (([], []), ([1], [1, 2]), ([0], [0]), ([math.nan], [1])):
        with pytest.raises(ValueError):
            cosine(first, second)


def test_nulls_and_errors_not_hidden() -> None:
    summary = summarize(
        [
            {"ok": True, "metrics": {"faithfulness": 0.5}},
            {"ok": False, "metrics": {"faithfulness": None}},
        ]
    )
    assert summary["errors"] == 1
    assert summary["metrics"]["faithfulness"] == {
        "mean": 0.5,
        "defined_cases": 1,
        "undefined_cases": 1,
    }


class Chat:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(self, system_prompt: str, prompt: str) -> ChatReply:
        return ChatReply(text="Давление составляет 10 МПа [S1].", finish_reason="stop")

    def close(self) -> None:
        pass


def setup(tmp_path: Path) -> tuple[RAGSettings, GoldenDataset]:
    (tmp_path / "source.txt").write_text("Давление составляет 10 МПа.", encoding="utf-8")
    settings = RAGSettings(
        document_root=tmp_path, qdrant_path=tmp_path / "qdrant", rerank_score_threshold=0.1
    )
    rows: list[dict[str, Any]] = [
        {
            "id": "one",
            "source": "source.txt",
            "source_url": "https://example.org/source",
            "question": "Давление?",
            "ground_truth_answer": "Давление составляет 10 МПа.",
            "quotes": ["Давление составляет 10 МПа."],
        }
    ]
    return settings, prepare(tmp_path, rows, settings)


def test_full_evaluation_and_trace(tmp_path: Path) -> None:
    settings, dataset = setup(tmp_path)
    with RAGPipeline(
        settings,
        retriever=HybridRetriever(HashingEncoder(64), LexicalReranker(), settings=settings),
        generator=LocalLLMGenerator(settings, backend=Chat()),
        fact_verifier=NLIFactVerifier(settings, backend=HeuristicNLI()),
    ) as pipeline:
        report = evaluate(pipeline, dataset, 3)
    assert report["summary"]["errors"] == 0
    metrics = report["rows"][0]["metrics"]
    for key in (
        "hybrid_hit_at_k",
        "reranked_mrr_at_k",
        "citation_precision",
        "citation_recall",
        "faithfulness",
    ):
        assert metrics[key] == 1
    assert report["rows"][0]["response"]["hybrid_candidate_ids"]
    assert report["dataset_status"] == "contains_ai_drafts_not_expert_validated"


def test_corpus_hash_mismatch_prevents_indexing(tmp_path: Path) -> None:
    settings, dataset = setup(tmp_path)
    (tmp_path / "source.txt").write_text("Different", encoding="utf-8")
    with RAGPipeline(settings) as pipeline:
        with pytest.raises(ValueError, match="hash mismatch"):
            evaluate(pipeline, dataset, 3)
        assert pipeline._retriever is None


def test_invalid_label_and_quote_rejected(tmp_path: Path) -> None:
    settings, dataset = setup(tmp_path)
    bad = dataset.model_dump()
    bad["examples"][0]["is_negative"] = True
    with pytest.raises(ValidationError):
        GoldenDataset.model_validate(bad)
    with pytest.raises(ValueError, match="Quote not found"):
        prepare(
            tmp_path,
            [
                {
                    "id": "bad",
                    "source": "source.txt",
                    "source_url": "https://example.org",
                    "quotes": ["invented"],
                }
            ],
            settings,
        )


def test_html_preserves_inline_words_and_removes_scripts() -> None:
    extractor = TextExtractor()
    extractor.feed("<h2>Section</h2><p>Some <em>important</em> words.</p><script>bad()</script>")
    assert "Some important words." in extractor.text()
    assert "## Section" in extractor.text() and "bad()" not in extractor.text()


def test_interactive_exit_without_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "/exit")
    assert cli_main(["--interactive", "--no-color"]) == 0
