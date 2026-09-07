"""Summary policy is independent of QA; deterministic models exercise all stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from document_parser import DocumentChunk
from generator_verifier import ChatReply, NLIProbabilities
from hybrid_retriever import HybridRetriever, LexicalReranker
from settings import RAGSettings
from pipeline import RAGPipeline
from summarization import (
    SummarizationPipeline,
    SummaryError,
    anti_contradiction,
    deduplicate_sentences,
    mmr_select,
)


def test_mmr_removes_duplicates_and_is_deterministic() -> None:
    vectors = [[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]]
    for _ in range(3):
        assert mmr_select(vectors, [1.0, 0.99, 0.8], limit=3) == [0, 2]
    assert mmr_select([[1, 0], [0.6, 0.8], [0, 1]], [0.9, 0.88, 0.4], limit=2, weight=0.4) == [0, 2]


@pytest.mark.parametrize(
    "entailment,neutral,contradiction,allowed",
    [
        (0.1, 0.8, 0.1, True),
        (0.8, 0.1, 0.1, True),
        (0.4, 0.3, 0.3, True),
        (0.1, 0.59, 0.31, False),
    ],
)
def test_anti_contradiction(
    entailment: float, neutral: float, contradiction: float, allowed: bool
) -> None:
    probabilities = NLIProbabilities(
        p_entailment=entailment, p_neutral=neutral, p_contradiction=contradiction
    )
    assert anti_contradiction(probabilities) is allowed


def test_post_dedup_cosine_and_invalid_vectors() -> None:
    assert deduplicate_sentences([[1, 0], [0.99, 0.02], [0, 1]]) == [0, 2]
    with pytest.raises(SummaryError, match="norm"):
        deduplicate_sentences([[0, 0]])
    with pytest.raises(SummaryError, match="NLI could not"):
        anti_contradiction(None)


class Encoder:
    dimension = 3

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0]
            if "alpha" in s.lower()
            else [0.0, 1.0, 0.0]
            if "beta" in s.lower()
            else [0.0, 0.0, 1.0]
            for s in texts
        ]


class NeutralNLI:
    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        return (
            NLIProbabilities(p_entailment=0.1, p_neutral=0.1, p_contradiction=0.8)
            if "99" in hypothesis
            else NLIProbabilities(p_entailment=0.1, p_neutral=0.8, p_contradiction=0.1)
        )


class Chat:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(self, system_prompt: str, prompt: str) -> ChatReply:
        self.calls += 1
        assert "технический аналитик" in system_prompt
        return ChatReply(text=self.text, finish_reason="stop")

    def close(self) -> None:
        pass


def config(tmp_path: Path) -> RAGSettings:
    return RAGSettings(mode="development", allow_fallback=True, qdrant_path=tmp_path / "index")


def chunk(i: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        str(i), text, "document", "manual.txt", (), None, None, (), 8, {"ordinal": i}
    )


def test_full_summary_filters_contradictions_and_semantic_duplicates(tmp_path: Path) -> None:
    settings = config(tmp_path)
    chat = Chat(
        "Alpha uses 10 units [S1].\n\n- Beta is supported [S1].\n- Alpha requires 10 units [S1].\n- Alpha uses 99 units [S1]."
    )
    with HybridRetriever(Encoder(), LexicalReranker(), settings=settings) as retriever:
        retriever.add([chunk(1, "Alpha uses 10 units. Beta is supported.")])
        with SummarizationPipeline(
            settings, retriever=retriever, chat=chat, nli=NeutralNLI()
        ) as pipeline:
            result = pipeline.summarize("document")
        assert not result.refused and chat.calls == 1
        assert result.summary == "Alpha uses 10 units [S1].\n\n- Beta is supported [S1]."
        assert [c.status for c in result.claims] == [
            "accepted",
            "accepted",
            "duplicate",
            "contradiction",
        ]
        assert result.claims[0].probabilities.p_neutral == 0.8
        assert retriever.size == 1  # Borrowed components are not closed.


class BudgetChat(Chat):
    def count_tokens(self, text: str) -> int:
        data = json.loads(text.splitlines()[-1])
        return sum(1500 if k.startswith("S") else 100 for k in data["sources"])

    def complete(self, system_prompt: str, prompt: str) -> ChatReply:
        self.calls += 1
        data = json.loads(prompt)
        parts = [
            f"{text} [{key}]." if key.startswith("S") else text
            for key, text in data["sources"].items()
        ]
        return ChatReply(text="\n\n".join(parts), finish_reason="stop")


def test_summary_reflows_verified_sentences_without_rewriting(tmp_path: Path) -> None:
    settings = config(tmp_path)
    chat = Chat("Alpha is specified [S1]. Beta is supported [S1].")
    with HybridRetriever(Encoder(), LexicalReranker(), settings=settings) as retriever:
        retriever.add([chunk(1, "Alpha is specified. Beta is supported.")])
        with SummarizationPipeline(
            settings, retriever=retriever, chat=chat, nli=NeutralNLI()
        ) as pipeline:
            result = pipeline.summarize("document")
    assert result.summary == "Alpha is specified [S1].\n\nBeta is supported [S1]."
    assert all(c.status == "accepted" for c in result.claims)


def test_budgeted_map_reduce_covers_every_selected_chunk(tmp_path: Path) -> None:
    settings = config(tmp_path)
    chat = BudgetChat("")
    with HybridRetriever(Encoder(), LexicalReranker(), settings=settings) as retriever:
        retriever.add(
            [chunk(i, f"{word} is specified") for i, word in enumerate(["Alpha", "Beta", "Gamma"])]
        )
        with SummarizationPipeline(
            settings, retriever=retriever, chat=chat, nli=NeutralNLI()
        ) as pipeline:
            result = pipeline.summarize("document")
        assert result.generation_calls == 3
        assert len(result.sources) == 3
        assert all(word in result.summary for word in ["Alpha", "Beta", "Gamma"])


def test_unknown_document_does_not_generate(tmp_path: Path) -> None:
    settings = config(tmp_path)
    chat = Chat("Must not generate")
    with HybridRetriever(Encoder(), LexicalReranker(), settings=settings) as retriever:
        with SummarizationPipeline(
            settings, retriever=retriever, chat=chat, nli=NeutralNLI()
        ) as pipeline:
            with pytest.raises(ValueError, match="not found"):
                pipeline.summarize("missing")
        assert chat.calls == 0


def test_cli_routes_around_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import baseline_rag
    import settings as settings_module
    import summarization

    settings = config(tmp_path)
    with HybridRetriever(Encoder(), LexicalReranker(), settings=settings) as retriever:
        retriever.add([chunk(1, "Alpha uses 10 units.")])
        pipeline = SummarizationPipeline(
            settings, retriever=retriever, chat=Chat("Alpha uses 10 units [S1]."), nli=NeutralNLI()
        )
        monkeypatch.setattr(settings_module, "RAGSettings", lambda: settings)
        monkeypatch.setattr(summarization, "SummarizationPipeline", lambda _: pipeline)
        monkeypatch.setattr(RAGPipeline, "from_environment", lambda: pytest.fail("QA route"))
        assert baseline_rag.main(["--summarize", "document", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["mode"] == "summarization"
