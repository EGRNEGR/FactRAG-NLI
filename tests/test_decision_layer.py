"""Decision layer (nli_decision=judge), recovery pass and the optimization_2 runtime switches."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from document_parser import DocumentChunk
from generator_verifier import (
    REFUSAL,
    ChatReply,
    ClaimJudge,
    GenerationMetadata,
    HeuristicNLI,
    LocalLLMGenerator,
    NLIFactVerifier,
    NLIProbabilities,
    RawGenerationResult,
    SourceContext,
    VerificationError,
    _HTTPBackend,
    binary_label_indices,
    judge_features,
    recovery_note,
)
from grounding import normalize_keywords
from hybrid_retriever import HashingEncoder, HybridRetriever, LexicalReranker
from model_runtime import runtime
from pipeline import RAGPipeline, recovery_due
from settings import RAGSettings

FEATURES = ("a_e_max", "b_e_max", "a_full_c", "anchors_ok")
# Support needs both models: logits add up, a missing anchor or a contradiction costs.
JUDGE = {
    "features": FEATURES,
    "coef": [1.0, 1.0, -4.0, 1.0],
    "intercept": -1.0,
    "threshold": 0.5,
    "partial_threshold": 0.25,
    "contradiction": 0.5,
    "disagreement": 0.6,
    "veto": 0.9,
}


def probs(e: float, c: float = 0.0) -> NLIProbabilities:
    return NLIProbabilities(p_entailment=e, p_neutral=1 - e - c, p_contradiction=c)


class TableNLI:
    """Probabilities chosen by a keyword of the hypothesis (same for every premise)."""

    def __init__(self, table: dict[str, NLIProbabilities]) -> None:
        self.table = table
        self.pairs = 0

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities:
        self.pairs += 1
        return next(p for key, p in self.table.items() if key in hypothesis)

    def predict_many(self, pairs: list[tuple[str, str]]) -> list[NLIProbabilities]:
        return [self.predict(p, h) for p, h in pairs]


def judge_settings(tmp_path: Path, **update: object) -> RAGSettings:
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(JUDGE), encoding="utf-8")
    return RAGSettings(
        mode="development",
        allow_fallback=True,
        nli_decision="judge",
        nli_judge_path=path,
        **update,
    )


CONTEXT = {
    "S1": SourceContext(
        chunk_id="c1",
        document_id="d",
        text="Давление в системе составляет 10 МПа. Температура не превышает 80 градусов.",
        citation="ГОСТ, 4.2",
        section_path=("4.2 Давление",),
    )
}


def verify(verifier: NLIFactVerifier, text: str) -> list[tuple[str, str]]:
    result = verifier.verify(
        RawGenerationResult(
            text=text,
            contexts=CONTEXT,
            metadata=GenerationMetadata(backend="t", finish_reason="stop"),
        )
    )
    return [(c.status.value, c.grade or "") for c in result.claims]


def test_two_way_labels_and_keyword_normalization() -> None:
    assert binary_label_indices({0: "entailment", 1: "not_entailment"}) == {
        "entailment": 0,
        "not_entailment": 1,
    }
    with pytest.raises(VerificationError):
        binary_label_indices({0: "entailment", 1: "neutral"})
    assert normalize_keywords("Cookie SHALL NOT contain it; MAY be set") == (
        "Cookie shall not contain it; may be set"
    )
    assert normalize_keywords("MUSTER of the MAYOR") == "MUSTER of the MAYOR"


def test_judge_features_and_probability() -> None:
    f = judge_features(
        [probs(0.2, 0.7), probs(0.9)],
        [probs(0.1), None],
        overlap=0.5,
        code_span=False,
        anchors_ok=True,
        has_anchors=True,
    )
    assert f["a_full_c"] == pytest.approx(0.7)
    assert f["a_e_max"] > 2 and f["b_e_max"] < 0  # logits of 0.9 and 0.1
    judge = ClaimJudge(JUDGE)
    assert 0 < judge.probability(f) < 1
    with pytest.raises(VerificationError):
        ClaimJudge({**JUDGE, "coef": [1.0]})


def test_judge_grades(tmp_path: Path) -> None:
    primary = TableNLI(
        {
            "10 МПа": probs(0.95),
            "20 МПа": probs(0.02, 0.95),
            "80": probs(0.97),
            "соседн": probs(0.4),
        }
    )
    secondary = TableNLI(
        {"10 МПа": probs(0.97), "20 МПа": probs(0.01), "80": probs(0.01), "соседн": probs(0.4)}
    )
    verifier = NLIFactVerifier(
        judge_settings(tmp_path, nli_co_verification=False),
        backend=primary,
        secondary=secondary,
    )
    assert verify(verifier, "Давление составляет 10 МПа [S1].") == [("verified", "supported")]
    assert verify(verifier, "Давление составляет 20 МПа [S1].")[0] == (
        "contradiction",
        "contradicted",
    )
    # mDeBERTa says yes, the second model says no: never published, graded ambiguous.
    assert verify(verifier, "Температура не превышает 80 градусов [S1].") == [
        ("unverified", "ambiguous")
    ]
    assert verify(verifier, "Это соседний факт [S1].") == [("unverified", "partially_supported")]
    assert secondary.pairs > 0


def test_judge_keeps_the_anchor_guard(tmp_path: Path) -> None:
    both = TableNLI({"": probs(0.99)})
    verifier = NLIFactVerifier(judge_settings(tmp_path), backend=both, secondary=both)
    # 15 occurs in no cited source: high entailment cannot publish it.
    assert verify(verifier, "Давление составляет 15 МПа [S1].")[0][0] == "unverified"


def test_judge_needs_its_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RAG_NLI_JUDGE_PATH"):
        RAGSettings(mode="development", allow_fallback=True, nli_decision="judge")
    with pytest.raises(VerificationError, match="secondary"):
        NLIFactVerifier(judge_settings(tmp_path), backend=HeuristicNLI())


def test_rules_decision_reports_grades() -> None:
    settings = RAGSettings(mode="development", allow_fallback=True)
    verifier = NLIFactVerifier(settings, backend=HeuristicNLI())
    assert verify(verifier, "Давление в системе составляет 10 МПа [S1].") == [
        ("verified", "supported")
    ]
    assert verify(verifier, "Давление 99 МПа [S1].")[0][1] == "insufficient_evidence"


class RecoveringChat:
    """Refuses unless asked with a recovery note or sampling; records every call."""

    supports_thinking = True

    def __init__(self, answer_on: str) -> None:
        self.answer_on = answer_on
        self.calls: list[dict[str, object]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        thinking: bool = False,
        sampling: dict[str, object] | None = None,
    ) -> ChatReply:
        noted = "повторный запрос" in system_prompt
        self.calls.append({"thinking": thinking, "noted": noted, "sampling": sampling})
        hit = (self.answer_on == "note" and noted) or (
            self.answer_on == "sampling" and sampling is not None and sampling.get("seed") == 1001
        )
        text = "Давление составляет 10 МПа [S1]." if hit else REFUSAL
        return ChatReply(text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=5)

    def close(self) -> None:
        pass


def recovery_pipeline(tmp_path: Path, chat: RecoveringChat, **update: object) -> RAGPipeline:
    root = tmp_path / "documents"
    root.mkdir(exist_ok=True)
    (root / "standard.md").write_text(
        "# ГОСТ\n## 4.2 Давление\nДавление составляет 10 МПа.", encoding="utf-8"
    )
    settings = RAGSettings(
        mode="development",
        allow_fallback=True,
        document_root=root,
        qdrant_path=tmp_path / "db",
        rerank_score_threshold=0.1,
        llm_think_retry="refusal",
        **update,
    )
    pipeline = RAGPipeline(
        settings,
        retriever=HybridRetriever(HashingEncoder(64), LexicalReranker(), settings=settings),
        generator=LocalLLMGenerator(settings, backend=chat),
        fact_verifier=NLIFactVerifier(settings, backend=HeuristicNLI()),
    )
    pipeline.index_documents([root / "standard.md"])
    return pipeline


@pytest.mark.parametrize(
    ("mode", "calls", "published"),
    [
        ("off", 2, False),  # draft + thinking pass
        ("evidence_prompt", 3, True),
        ("sampling", 4, True),  # the second sample answers
    ],
)
def test_recovery_after_a_refused_thinking_pass(
    tmp_path: Path, mode: str, calls: int, published: bool
) -> None:
    chat = RecoveringChat("note" if mode == "evidence_prompt" else "sampling")
    with recovery_pipeline(tmp_path, chat, llm_recovery=mode, llm_recovery_min_rerank=0.0) as p:
        response = p.query("Какое давление?")
    assert len(chat.calls) == calls == len(response.verification_attempts)
    assert [c["thinking"] for c in chat.calls[:2]] == [False, True]
    assert (response.refusal_reason is None) is published
    flags = [a["generation_metadata"]["recovery"] for a in response.verification_attempts]
    assert flags == [False, False] + [True] * (calls - 2)
    if mode == "sampling":
        assert [c["sampling"] for c in chat.calls[2:]] == [
            {"temperature": 0.7, "seed": 1000},
            {"temperature": 0.7, "seed": 1001},
        ]


def test_recovery_needs_a_relevant_source(tmp_path: Path) -> None:
    chat = RecoveringChat("note")
    with recovery_pipeline(
        tmp_path, chat, llm_recovery="evidence_prompt", llm_recovery_min_rerank=1.0
    ) as p:
        response = p.query("Какое давление?")
    assert len(chat.calls) == 2 and response.refusal_reason == "llm_refusal"


def test_recovery_trigger_and_note() -> None:
    settings = RAGSettings(mode="development", allow_fallback=True, llm_recovery="sampling")
    assert not recovery_due(settings, "nli_verification_failed", [])
    assert not recovery_due(settings, "llm_refusal", [])
    note = recovery_note(["S2", "S1"])
    assert "S2, S1" in note and REFUSAL in note


def test_sampling_overrides_reach_the_request() -> None:
    settings = RAGSettings(
        mode="development",
        allow_fallback=True,
        llm_api_base="http://127.0.0.1:8080/v1",
        llm_api_model="qwen",
        llm_disable_thinking=True,
    )
    generator = LocalLLMGenerator(settings)
    backend = generator._backend
    assert isinstance(backend, _HTTPBackend)
    backend.client.close()
    bodies: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "10 МПа [S1]."}, "finish_reason": "stop"}]},
        )

    backend.client = httpx.Client(
        base_url="http://127.0.0.1:8080/v1/", transport=httpx.MockTransport(handle)
    )
    try:
        chunk = DocumentChunk("one", "10 МПа", "doc", "source.txt", (), None, None, (), 3)
        generator.generate("Давление?", [chunk])
        generator.generate(
            "Давление?", [chunk], sampling={"temperature": 0.7, "seed": 7}, system_note="NOTE"
        )
        assert [(b["temperature"], b.get("seed")) for b in bodies] == [(0.0, None), (0.7, 7)]
        assert bodies[1]["messages"][0]["content"].endswith("\nNOTE")  # type: ignore[index]
        assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        generator.close()


def test_embedding_device_overrides_model_device() -> None:
    import torch

    settings = RAGSettings(mode="development", allow_fallback=True, embedding_device="cpu")
    assert runtime(settings, device=settings.embedding_device) == ("cpu", torch.float32)
    with pytest.raises(ValueError):
        RAGSettings(mode="development", allow_fallback=True, embedding_device="tpu")


class RevisingChat:
    """Wrong without thinking, right with it; records whether each prompt carried a revision."""

    supports_thinking = True

    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        thinking: bool = False,
        sampling: dict[str, object] | None = None,
    ) -> ChatReply:
        revision = json.loads(prompt).get("revision")
        self.calls.append((thinking, revision is not None))
        if revision is not None:
            assert revision["rejected_claims"][0]["text"].startswith("Давление составляет 99")
        text = (
            "Давление составляет 10 МПа [S1]." if thinking else "Давление составляет 99 МПа [S1]."
        )
        return ChatReply(text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=5)

    def close(self) -> None:
        pass


@pytest.mark.parametrize(("mode", "revised"), [("fresh", False), ("revise", True)])
def test_thinking_pass_revises_a_rejected_answer(tmp_path: Path, mode: str, revised: bool) -> None:
    chat = RevisingChat()
    with recovery_pipeline(tmp_path, chat, llm_think_mode=mode) as pipeline:  # type: ignore[arg-type]
        response = pipeline.query("Какое давление?")
    assert chat.calls == [(False, False), (True, revised)]
    assert response.refusal_reason is None
