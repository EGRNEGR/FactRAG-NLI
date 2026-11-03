"""query_translation: a question refused in another script than its sources is translated,
retrieved again and drafted once; the refusal stays unless the new answer is published."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generator_verifier import REFUSAL, ChatReply, HeuristicNLI, LocalLLMGenerator, NLIFactVerifier
from hybrid_retriever import HashingEncoder, HybridRetriever, LexicalReranker
from pipeline import RAGPipeline
from query_translation import parse_rewrite, text_script, translation_prompt
from settings import RAGSettings


class TranslatingChat:
    """Translates into Russian; answers Russian questions, refuses English ones."""

    supports_thinking = True

    def __init__(self, translation: str, answer: str = "Давление составляет 10 МПа [S1].") -> None:
        self.translation = translation
        self.answer = answer
        self.rewrites: list[tuple[str, str]] = []
        self.questions: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        thinking: bool = False,
        sampling: dict[str, object] | None = None,
    ) -> ChatReply:
        assert not thinking
        if system_prompt.startswith("Переведи"):
            self.rewrites.append((system_prompt, prompt))
            text = self.translation
        else:
            question = json.loads(prompt)["question"]
            self.questions.append(question)
            text = self.answer if text_script(question) == "ru" else REFUSAL
        return ChatReply(text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=5)

    def close(self) -> None:
        pass


def pipeline(tmp_path: Path, chat: TranslatingChat, mode: str) -> RAGPipeline:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "standard.md").write_text(
        "# ГОСТ\n## 4.2 Режим\nДавление составляет 10 МПа. Температура составляет 300 градусов.",
        encoding="utf-8",
    )
    settings = RAGSettings(
        mode="development",
        allow_fallback=True,
        document_root=root,
        qdrant_path=tmp_path / "db",
        rerank_score_threshold=0.1,
        query_translation=mode,
    )
    p = RAGPipeline(
        settings,
        retriever=HybridRetriever(HashingEncoder(64), LexicalReranker(), settings=settings),
        generator=LocalLLMGenerator(settings, backend=chat),
        fact_verifier=NLIFactVerifier(settings, backend=HeuristicNLI()),
    )
    p.index_documents([root / "standard.md"])
    return p


def test_script_and_reply_parsing() -> None:
    assert text_script("What is the pressure?") == "en"
    assert text_script("Какое давление в SP 800-53?") == "ru"
    assert text_script("10 / 20") is None
    assert "русский язык" in translation_prompt("ru")
    assert parse_rewrite("«Какое давление?»\nПояснение", "What is the pressure?") == (
        "Какое давление?"
    )
    assert parse_rewrite("  \n", "What?") is None
    assert parse_rewrite("x" * 200, "What?") is None  # an answer, not a translation


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("What is the pressure?", "relevance_threshold_not_met"),  # no Russian term in common
        ("What is the 10 МПа value?", "llm_refusal"),  # passes the gate, the LLM refuses
    ],
)
def test_refusal_is_answered_in_the_sources_language(
    tmp_path: Path, question: str, reason: str
) -> None:
    chat = TranslatingChat("Какое давление?")
    with pipeline(tmp_path, chat, "refusal") as p:
        response = p.query(question)
    assert len(chat.rewrites) == 1 and chat.rewrites[0][1] == question
    assert "русский язык" in chat.rewrites[0][0]
    assert not response.refused and "10 МПа" in response.answer
    assert response.query == question
    assert response.translation is not None
    assert response.translation["question"] == "Какое давление?"
    assert response.translation["target"] == "ru"
    assert response.translation["refusal_reason"] is None
    assert response.sources and "Давление составляет 10 МПа" in response.sources[0].text
    assert chat.questions[-1] == "Какое давление?"
    assert response.execution_stats["translation_ms"] >= 0
    # the refused attempt of the original question stays in the log before the new draft
    expected = 2 if reason == "llm_refusal" else 1
    assert len(response.verification_attempts) == expected


def test_off_keeps_the_refusal(tmp_path: Path) -> None:
    chat = TranslatingChat("Какое давление?")
    with pipeline(tmp_path, chat, "off") as p:
        response = p.query("What is the pressure?")
    assert response.refusal_reason == "relevance_threshold_not_met"
    assert chat.rewrites == [] and response.translation is None


def test_question_in_the_sources_language_is_not_translated(tmp_path: Path) -> None:
    chat = TranslatingChat("unused", answer=REFUSAL)
    with pipeline(tmp_path, chat, "refusal") as p:
        response = p.query("Какое давление?")
    assert response.refusal_reason == "llm_refusal"
    assert chat.rewrites == [] and response.translation is None


def test_refused_translation_keeps_the_original_refusal(tmp_path: Path) -> None:
    chat = TranslatingChat("Какое давление?", answer=REFUSAL)
    with pipeline(tmp_path, chat, "refusal") as p:
        response = p.query("What is the 10 МПа value?")
    assert response.refused and response.refusal_reason == "llm_refusal"
    assert response.query == "What is the 10 МПа value?"
    assert response.translation is not None
    assert response.translation["refusal_reason"] == "llm_refusal"
    assert len(response.verification_attempts) == 2  # both drafts are logged


def test_published_answer_is_not_translated(tmp_path: Path) -> None:
    chat = TranslatingChat("unused")
    with pipeline(tmp_path, chat, "refusal") as p:
        response = p.query("Какое давление?")
    assert not response.refused and chat.rewrites == [] and response.translation is None
