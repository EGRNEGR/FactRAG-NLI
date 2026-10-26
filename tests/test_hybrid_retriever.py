"""Use real embedded Qdrant/BM25; explicit lexical adapters test orchestration only."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence
from types import SimpleNamespace
import sys

import pytest

from document_parser import DocumentChunk, ParsedChunk
from hybrid_retriever import (
    HashingEncoder,
    HybridRetriever,
    LexicalReranker,
    RetrievalError,
    reciprocal_rank_fusion,
    tokenize,
    SentenceTransformerEncoder,
    CrossEncoderReranker,
    heading_only,
)
from settings import RAGSettings


def configuration(path: Path, **overrides: object) -> RAGSettings:
    values: dict[str, object] = {
        "mode": "development",
        "allow_fallback": True,
        "qdrant_path": path,
        "rerank_score_threshold": 0.1,
    }
    values.update(overrides)
    return RAGSettings.model_validate(values)


def test_navigation_title_is_not_factual_evidence() -> None:
    title = replace(chunk("title", "Hot Standby"), section_path=("27.4 Hot Standby",))
    assert heading_only(title)
    assert not heading_only(replace(title, text="Hot Standby accepts read-only queries."))
    assert not heading_only(replace(title, text="10 MPa"))


def test_heading_cannot_displace_evidence(tmp_path: Path) -> None:
    with retriever(tmp_path / "headings", rerank_score_threshold=0) as search:
        title = replace(chunk("title", "Hot Standby"), section_path=("27.4 Hot Standby",))
        body = replace(title, chunk_id="body", text="Hot Standby accepts read-only queries.")
        search.add([title, body])
        assert [r.chunk.chunk_id for r in search.search("Hot Standby")] == ["body"]


def chunk(identifier: str, text: str = "давление 10 МПа", document: str = "gost") -> ParsedChunk:
    return DocumentChunk(
        identifier,
        text,
        document,
        "gost.pdf",
        ("ГОСТ", "4.2"),
        3,
        4,
        ("table-1",),
        10,
        {
            "table_context": [{"table_number": "1", "columns": ["Параметр", "Значение"]}],
            "footnote_context": ["При 20 °C"],
        },
    )


def retriever(path: Path, **overrides: object) -> HybridRetriever:
    return HybridRetriever(
        HashingEncoder(64), LexicalReranker(), settings=configuration(path, **overrides)
    )


def test_weighted_rrf_formula_and_ties() -> None:
    result = reciprocal_rank_fusion(
        {"dense": ["b", "a"], "sparse": ["a", "c"]}, weights={"dense": 2.0, "sparse": 0.5}, k=60
    )
    assert result["a"] == pytest.approx(2 / 62 + 0.5 / 61)
    assert result["b"] == pytest.approx(2 / 61)
    assert result["c"] == pytest.approx(0.5 / 62)
    ties = reciprocal_rank_fusion({"sparse": ["b"], "dense": ["a"]})
    assert list(ties) == ["a", "b"]
    assert ties == reciprocal_rank_fusion({"dense": ["a"], "sparse": ["b"]})


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_rrf_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["a"]}, weights={"dense": weight})


def test_rrf_duplicates_and_zero_weight() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        reciprocal_rank_fusion({"dense": ["a", "a"]})
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({}, k=0)
    assert reciprocal_rank_fusion({"dense": ["a"]}, weights={"dense": 0.0}) == {}


def test_tokenizer_keeps_negation_units_numbers() -> None:
    assert tokenize("НЕ более 10 МПа; The pressure is NOT zero. Ёмкость") == [
        "не",
        "более",
        "10",
        "мпа",
        "pressure",
        "not",
        "zero",
        "емкость",
    ]


def test_real_qdrant_payload_and_restart(tmp_path: Path) -> None:
    with retriever(tmp_path) as index:
        assert index.add([chunk("one")]) == 1
        hit = index.search("давление")[0]
        assert hit.chunk.page_start == 3 and hit.chunk.page_end == 4
        assert hit.chunk.metadata["table_context"][0]["table_number"] == "1"
        assert hit.chunk.metadata["footnote_context"] == ["При 20 °C"]
        assert hit.sources == ("dense", "sparse")
        assert hit.rrf_score == hit.score == pytest.approx(2 / 61)
        assert hit.dense_score is not None and hit.sparse_score is not None
        assert hit.rerank_score == 1
    with retriever(tmp_path) as reopened:
        hit = reopened.search("давление")[0]
        assert reopened.size == 1
        assert hit.chunk.metadata["table_context"][0]["columns"] == ["Параметр", "Значение"]
        assert hit.chunk.section_path == ("ГОСТ", "4.2")


@pytest.mark.parametrize("damage", ["missing", "corrupt", "stale"])
def test_cache_recovery(tmp_path: Path, damage: str) -> None:
    with retriever(tmp_path) as index:
        index.add([chunk("one")])
    cache = tmp_path / "rag_chunks.bm25.json"
    if damage == "missing":
        cache.unlink()
    else:
        cache.write_text("{" if damage == "corrupt" else json.dumps({"version": 0, "ids": []}))
    with retriever(tmp_path) as recovered:
        assert recovered.search("давление")[0].chunk.chunk_id == "one"
    assert json.loads(cache.read_text(encoding="utf-8"))["ids"] == ["one"]


def test_empty_query_empty_index_and_no_match(tmp_path: Path) -> None:
    with retriever(tmp_path) as index:
        with pytest.raises(ValueError):
            index.search(" \n ")
        assert index.search("давление") == []
        index.add([chunk("one")])
        assert index.search("электропроводность") == []
        with pytest.raises(ValueError):
            index.search("давление", top_k=0)


def test_all_filtered_and_threshold_boundary(tmp_path: Path) -> None:
    with retriever(tmp_path, rerank_score_threshold=1.0) as index:
        index.add([chunk("one")])
        assert index.search("давление")
        assert index.search("давление температура") == []


def test_deterministic_ties_and_upsert(tmp_path: Path) -> None:
    with retriever(tmp_path) as index:
        index.add([chunk("b"), chunk("a")])
        assert [hit.chunk.chunk_id for hit in index.search("давление")] == ["a", "b"]
        index.add([chunk("a", "температура 20 C")])
        assert index.size == 2
        assert [hit.chunk.chunk_id for hit in index.search("температура")] == ["a"]


def test_delete_and_restart(tmp_path: Path) -> None:
    with retriever(tmp_path) as index:
        index.add([chunk("one"), chunk("two", document="other")])
        assert index.delete_document("gost") == 1
        assert index.delete_document("absent") == 0
    with retriever(tmp_path) as index:
        assert [hit.chunk.document_id for hit in index.search("давление")] == ["other"]
        assert index.delete_document("other") == 1
        assert index.search("давление") == []


def test_scroll_more_than_one_page(tmp_path: Path) -> None:
    with retriever(tmp_path) as index:
        index.add([chunk(str(i)) for i in range(270)])
    with retriever(tmp_path) as index:
        assert index.size == 270


def test_dimension_change_rejected(tmp_path: Path) -> None:
    with retriever(tmp_path) as index:
        index.add([chunk("one")])
    with pytest.raises(RetrievalError, match="dimension"):
        HybridRetriever(HashingEncoder(128), LexicalReranker(), settings=configuration(tmp_path))


def test_production_missing_models_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODE", "production")
    monkeypatch.setenv("RAG_ALLOW_FALLBACK", "false")
    monkeypatch.setenv("RAG_API_TOKENS", '["01234567890123456789012345678901"]')
    monkeypatch.setenv("RAG_EMBEDDING_MODEL_PATH", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="directory does not exist"):
        HybridRetriever()


def test_explicit_fallback_is_reported(tmp_path: Path) -> None:
    with HybridRetriever(
        settings=configuration(
            tmp_path,
            embedding_model_path=tmp_path / "absent",
            reranker_model_path=tmp_path / "absent",
        )
    ) as index:
        assert isinstance(index.encoder, HashingEncoder)
        assert isinstance(index.reranker, LexicalReranker)
        assert len(index.fallback_reasons) == 2


class InvalidEncoder(HashingEncoder):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[math.nan] * self.dimension for _ in texts]


def test_invalid_vectors_do_not_mutate_storage(tmp_path: Path) -> None:
    with HybridRetriever(
        InvalidEncoder(64), LexicalReranker(), settings=configuration(tmp_path)
    ) as index:
        with pytest.raises(RetrievalError, match="invalid vector"):
            index.add([chunk("one")])
        assert index.size == 0


def test_non_json_payload_and_duplicates_rejected(tmp_path: Path) -> None:
    with retriever(tmp_path) as index:
        with pytest.raises(ValueError, match="Duplicate"):
            index.add([chunk("one"), chunk("one")])
        with pytest.raises(Exception):
            index.add([replace(chunk("one"), metadata={"bad": object()})])
        assert index.size == 0


def test_close_releases_lock_and_blocks_queries(tmp_path: Path) -> None:
    index = retriever(tmp_path)
    index.close()
    index.close()
    with pytest.raises(RetrievalError, match="closed"):
        index.search("давление")
    with retriever(tmp_path) as reopened:
        assert reopened.size == 0


def test_cache_write_failure_requires_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = retriever(tmp_path)

    def fail(state: dict[str, object]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(index, "_write_cache", fail)
    with pytest.raises(RetrievalError, match="mutation failed"):
        index.add([chunk("one")])
    with pytest.raises(RetrievalError, match="failed"):
        index.search("давление")
    index.close()
    with retriever(tmp_path) as reopened:
        assert reopened.search("давление")[0].chunk.chunk_id == "one"


class InvalidReranker:
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [math.nan] * len(texts)


def test_nonfinite_reranker_is_rejected(tmp_path: Path) -> None:
    with HybridRetriever(
        HashingEncoder(64), InvalidReranker(), settings=configuration(tmp_path)
    ) as index:
        index.add([chunk("one")])
        with pytest.raises(RetrievalError, match="finite probability"):
            index.search("давление")


def artifact_layout(path: Path) -> None:
    """Artificial layout for loader argument tests, never an inference checkpoint."""
    path.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "modules.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors").write_bytes(b"test-layout-only")


def test_embedding_loader_receives_local_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model"
    artifact_layout(path)
    captured: dict[str, object] = {}

    class StopAtConstructor:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            captured.update(kwargs)
            captured["model_path"] = model_path
            raise RuntimeError("loader boundary reached")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=StopAtConstructor)
    )
    with pytest.raises(RuntimeError, match="loader boundary"):
        SentenceTransformerEncoder(configuration(tmp_path / "db", embedding_model_path=path))
    assert captured["local_files_only"] is True
    assert captured["trust_remote_code"] is False
    assert captured["model_path"] == str(path)
    assert captured["tokenizer_kwargs"] == {"local_files_only": True}


def test_incomplete_classifier_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model"
    artifact_layout(path)
    captured: list[dict[str, object]] = []

    class TokenizerBoundary:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            captured.append(kwargs)
            return object()

    class ModelBoundary:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> tuple[object, dict[str, list[str]]]:
            captured.append(kwargs)
            return object(), {"missing_keys": ["classifier.weight"]}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=TokenizerBoundary, AutoModelForSequenceClassification=ModelBoundary
        ),
    )
    with pytest.raises(RetrievalError, match="randomly initialized"):
        CrossEncoderReranker(configuration(tmp_path / "db", reranker_model_path=path))
    assert all(options["local_files_only"] is True for options in captured)
    assert all(options["trust_remote_code"] is False for options in captured)
