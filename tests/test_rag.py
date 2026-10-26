from __future__ import annotations

from document_parser import ChunkingConfig, DocumentParser
from generator_verifier import (
    GeneratorVerifier,
    HeuristicNLI,
    NLIStatus,
    LocalLLMGenerator,
    ChatReply,
    REFUSAL,
)
from hybrid_retriever import HashingEncoder, HybridRetriever, LexicalReranker, RetrieverConfig
from pipeline import RAGPipeline


class FixedGenerator:
    def __init__(self, response: str) -> None:
        self.response = response

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(self, system_prompt: str, prompt: str) -> ChatReply:
        return ChatReply(text=self.response, finish_reason="stop")

    def close(self) -> None:
        return None


def test_parser_preserves_hierarchy_and_table() -> None:
    text = """# ГОСТ 1\n\n## Раздел 4\n\n### 4.2.1 Требования\n\nДопустимое давление составляет 10 МПа.\n\n| Параметр | Значение |\n| Давление | 10 МПа |\n\nПримечание — Значение проверяют ежегодно.\n"""
    parsed = DocumentParser(ChunkingConfig(target_tokens=20, max_tokens=80)).parse_text(
        text, source="gost.md"
    )
    assert parsed.elements[2].section_path == ("ГОСТ 1", "Раздел 4", "4.2.1 Требования")
    table = next(element for element in parsed.elements if element.kind == "table")
    assert table.text.startswith("| Параметр | Значение |")
    assert "| --- | --- |" in table.text
    assert any("4.2.1 Требования" in chunk.citation for chunk in parsed.chunks)


def test_hybrid_retriever_combines_dense_and_sparse() -> None:
    pipeline = RAGPipeline(
        retriever=HybridRetriever(
            HashingEncoder(64),
            LexicalReranker(),
            config=RetrieverConfig(rerank_threshold=0.0, top_k=5),
        )
    )
    pipeline.ingest_text("# Раздел 4\n\nДавление в системе составляет 10 МПа.", source="gost.md")
    results = pipeline.retriever.search("Какое давление в системе?")
    assert results
    assert "давление" in results[0].chunk.text.lower()
    assert "sparse" in results[0].sources or "dense" in results[0].sources


def test_verifier_blocks_neutral_claims() -> None:
    pipeline = RAGPipeline(
        verifier=GeneratorVerifier(
            LocalLLMGenerator(
                backend=FixedGenerator(
                    "Давление в системе составляет 10 МПа. [S1]\nНорматив требует 99 лет. [S1]"
                )
            ),
            HeuristicNLI(),
        )
    )
    pipeline.ingest_text("# Раздел 4\n\nДавление в системе составляет 10 МПа.", source="gost.md")
    response = pipeline.query("Какое давление в системе?")
    assert "10 МПа" in response.answer
    assert "99 лет" not in response.answer
    assert any(
        claim.status in {NLIStatus.NEUTRAL.value, NLIStatus.CONTRADICTION.value}
        for claim in response.claims
    )


def test_pipeline_empty_index_is_safe() -> None:
    pipeline = RAGPipeline()
    response = pipeline.query("Что известно?")
    assert response.answer == REFUSAL
