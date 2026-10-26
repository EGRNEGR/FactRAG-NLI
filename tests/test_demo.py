"""Demo integration with explicit diagnostic inference, no model downloads."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

import demo
from test_pipeline import RecordedChat, configured, document, make_pipeline


def test_index_once_and_export_same_query(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    document(settings)
    chat = RecordedChat()
    with make_pipeline(settings, chat) as pipeline:
        assert demo.ensure_index(pipeline)["action"] == "indexed"
        count = pipeline.retriever.size
        assert demo.ensure_index(pipeline) == {"action": "reused", "chunks": count}
        response = pipeline.query("Какое давление?")
        snapshot = demo.retrieval_snapshot(pipeline, response)
        assert chat.calls == 1
        assert snapshot[0]["chunk_id"] == response.reranked_candidate_ids[0]
        assert snapshot[0]["source_id"] == "S1"
        assert snapshot[0]["rrf_score"] > 0
        out, html = demo.export_demo(tmp_path / "demo.json", response, snapshot)
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["response"]["answer"] == chat.answer
        assert saved["retrieved_top5"] == snapshot
        assert "ground_truth_answer" not in saved["response"]
        assert "ENTAILED" in html.read_text(encoding="utf-8")
    assert chat.closed


def test_export_escapes_untrusted_draft(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with make_pipeline(settings, RecordedChat()) as pipeline:
        response = pipeline.query("давление")
        response = response.model_copy(
            update={"raw_answer": "<script>alert(1)</script>\x1b[2J[red]draft"}
        )
        _, html = demo.export_demo(tmp_path / "unsafe.json", response, [])
        content = html.read_text(encoding="utf-8")
        assert "<script>" not in content
        assert "&lt;script&gt;" in content
        assert "\x1b" not in content
        assert "NEUTRAL" not in content


def test_empty_document_root_fails(tmp_path: Path) -> None:
    with make_pipeline(configured(tmp_path), RecordedChat()) as pipeline:
        with pytest.raises(ValueError, match="Нет документов"):
            demo.ensure_index(pipeline)


def test_presets_only_show_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "RAGPipeline", lambda *a: pytest.fail("Models must not load"))
    assert demo.main(["--list-presets", "--no-color"]) == 0
    assert demo.main(["--preset", "missing-id"]) == 1


def test_arbitrary_query_does_not_load_qa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = configured(tmp_path)
    pipeline = make_pipeline(settings, RecordedChat())
    monkeypatch.setattr(demo, "load_presets", lambda *a: pytest.fail("QA must not load"))
    monkeypatch.setattr(demo, "RAGPipeline", lambda *a: pipeline)
    assert demo.main(["--query", "давление", "--no-color"]) == 0


def test_no_evidence_is_not_neutral(tmp_path: Path) -> None:
    with make_pipeline(configured(tmp_path), RecordedChat()) as pipeline:
        response = pipeline.query("давление")
        output = io.StringIO()
        demo.render(Console(file=output, width=110), response, [])
        assert "Утверждений для NLI нет" in output.getvalue()
        assert "NEUTRAL" not in output.getvalue()
