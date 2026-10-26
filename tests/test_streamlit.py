"""Мы проверяем наш UI через AppTest без запуска моделей и рабочего индекса."""

from pathlib import Path
from typing import Any, Callable

import pytest
from streamlit.testing.v1 import AppTest

import web_bridge


class FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.fail = False

    def documents(self) -> list[web_bridge.DocumentInfo]:
        return [web_bridge.DocumentInfo("doc-1", "manual.txt", 2)]

    def run(
        self, question: str, mode: str, document_id: str | None, on_status: Callable[[str], None]
    ) -> dict[str, Any]:
        self.calls.append((question, mode, document_id))
        on_status("Мы проверяем факты")
        if self.fail:
            raise RuntimeError("private-token-path")
        return {
            "answer": "Наш проверенный ответ [S1].",
            "mode": mode,
            "sources": [{"source": "manual.txt", "text": "Факт", "rrf_score": 0.03}],
            "claims": [{"text": "Факт", "status": "verified"}],
            "execution_stats": {"generation_ms": 1500.0},
            "refused": False,
            "policy": "",
        }


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, FakeBridge]:
    bridge = FakeBridge()
    monkeypatch.setattr(web_bridge, "get_bridge", lambda: bridge)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
    app.run()
    assert not app.exception
    return app, bridge


def test_modes_change_state_and_placeholder_without_loading_models(
    ui: tuple[AppTest, FakeBridge],
) -> None:
    app, bridge = ui
    initial = app.chat_input[0].placeholder
    app.radio(key="mode").set_value("summary").run()
    assert not app.exception
    assert app.session_state["mode"] == "summary"
    assert app.session_state["input_placeholder"] != initial
    assert app.chat_input[0].placeholder == app.session_state["input_placeholder"]
    assert app.selectbox(key="summary_document_id").value == "doc-1"
    app.radio(key="mode").set_value("qa").run()
    assert app.chat_input[0].placeholder == initial
    assert not bridge.calls


def test_verified_qa_answer_metadata_and_history(ui: tuple[AppTest, FakeBridge]) -> None:
    app, bridge = ui
    app.chat_input[0].set_value("Наш вопрос").run()
    assert not app.exception
    assert bridge.calls == [("Наш вопрос", "qa", None)]
    assert app.session_state["messages"][-1]["content"] == "Наш проверенный ответ [S1]."
    assert len(app.expander) == 3
    app.run()
    assert len(bridge.calls) == 1
    assert any("1.500 с" in str(item.value) for item in app.markdown)


def test_summary_routes_by_document_name(ui: tuple[AppTest, FakeBridge]) -> None:
    app, bridge = ui
    app.radio(key="mode").set_value("summary").run()
    app.chat_input[0].set_value("manual.txt").run()
    assert not app.exception
    assert bridge.calls == [("manual.txt", "summary", "doc-1")]


def test_summary_unknown_name_does_not_run_selected_document(
    ui: tuple[AppTest, FakeBridge],
) -> None:
    app, bridge = ui
    app.radio(key="mode").set_value("summary").run()
    app.chat_input[0].set_value("unknown.txt").run()
    assert not app.exception and not bridge.calls
    assert app.error


def test_failures_hide_private_exception_and_allow_retry(ui: tuple[AppTest, FakeBridge]) -> None:
    app, bridge = ui
    bridge.fail = True
    app.chat_input[0].set_value("Наш вопрос").run()
    assert not app.exception
    assert "private-token-path" not in str(app.session_state["messages"])
    bridge.fail = False
    app.chat_input[0].set_value("Повторный вопрос").run()
    assert not app.exception
    assert len(bridge.calls) == 2


def test_new_session_has_no_previous_chat(ui: tuple[AppTest, FakeBridge]) -> None:
    app, _ = ui
    app.chat_input[0].set_value("Наша первая сессия").run()
    second = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not second.exception
    assert "Наша первая сессия" not in str(second.session_state["messages"])
