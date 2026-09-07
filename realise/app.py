"""Streamlit UI for local RAG web bridge."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypedDict, cast

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

Mode = Literal["qa", "summary"]


class BridgeResult(TypedDict, total=False):
    answer: str
    mode: str
    sources: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    execution_stats: dict[str, float]
    refused: bool
    policy: str


class WebBridge(Protocol):
    def documents(self) -> list[Any]: ...

    def run(
        self,
        question: str,
        mode: Mode,
        document_id: str | None,
        on_status: Callable[[str], None],
    ) -> BridgeResult: ...

    def ingest(
        self,
        name: str,
        data: bytes,
        on_status: Callable[[str], None],
    ) -> str: ...


class ChatMessage(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str
    mode: Mode
    timestamp: float
    meta: BridgeResult | None


logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Наши документы · RAG",
    page_icon="🧠",
    layout="wide",
)


@dataclass(frozen=True)
class DocumentChoice:
    document_id: str
    name: str
    chunks: int


def _safe_mode(value: str | None) -> Mode:
    return "summary" if value == "summary" else "qa"


def _mode_label(mode: Mode) -> str:
    return "Мы ищем ответ" if mode == "qa" else "Мы составляем саммари"


def _mode_placeholder(mode: Mode) -> str:
    return (
        "Какой факт мы найдём в наших документах?"
        if mode == "qa"
        else "По какому документу мы составим саммари? Введём название или ID…"
    )


def _default_doc_label(doc: DocumentChoice) -> str:
    return f"{doc.name} · {doc.document_id[:8]} · фрагментов: {doc.chunks}"


def _extract_field(document: Any, field_name: str, default: Any = "") -> Any:
    if isinstance(document, dict):
        return document.get(field_name, default)
    return getattr(document, field_name, default)


def _normalize_documents(raw_documents: list[Any]) -> list[DocumentChoice]:
    normalized: list[DocumentChoice] = []
    for item in raw_documents:
        document_id = str(_extract_field(item, "document_id", "")).strip()
        if not document_id:
            continue
        name = str(_extract_field(item, "name", document_id)).strip() or document_id
        chunks_raw = _extract_field(item, "chunks", 0)
        chunks = int(chunks_raw) if isinstance(chunks_raw, int) and chunks_raw >= 0 else 0
        normalized.append(DocumentChoice(document_id=document_id, name=name, chunks=chunks))
    return normalized


def _load_bridge() -> tuple[WebBridge | None, str | None]:
    try:
        from web_bridge import get_bridge

        return cast(WebBridge, get_bridge()), None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize web_bridge: %s", exc)
        return None, "Мы не подключились к нашему контуру. Проверим настройки и локальный сервер."


def _initialize_state() -> None:
    st.session_state.setdefault("mode", "qa")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault(
        "input_placeholder",
        _mode_placeholder(_safe_mode(st.session_state.get("mode"))),
    )
    st.session_state.setdefault("summary_document_id", "")
    st.session_state.setdefault("summary_search", "")
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("bridge_error", None)


def _sync_placeholder() -> None:
    mode = _safe_mode(st.session_state.get("mode"))
    st.session_state["mode"] = mode
    st.session_state["input_placeholder"] = _mode_placeholder(mode)


def _load_documents(bridge: WebBridge | None) -> list[DocumentChoice]:
    if bridge is None:
        return []
    try:
        return _normalize_documents(bridge.documents())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list documents: %s", exc)
        return []


def _push_message(message: ChatMessage) -> None:
    messages: list[ChatMessage] = cast(list[ChatMessage], st.session_state["messages"])
    messages.append(message)
    st.session_state["messages"] = messages[-40:]


def _display_message(message: ChatMessage) -> None:
    if message["role"] not in {"user", "assistant"}:
        return
    container = st.chat_message(message["role"])
    with container:
        container.markdown(message.get("content", ""))


def _display_metadata(meta: BridgeResult) -> None:
    sources = meta.get("sources", [])
    claims = meta.get("claims", [])
    execution_stats = meta.get("execution_stats", {})
    refused = bool(meta.get("refused", False))

    with st.expander(
        "Наш извлечённый контекст · MMR"
        if meta.get("mode") == "summary"
        else "Наш извлечённый контекст · Top-5 / RRF",
        expanded=False,
    ):
        if sources:
            for idx, source in enumerate(
                sources if meta.get("mode") == "summary" else sources[:5], start=1
            ):
                text = str(source.get("text", "")).strip()
                score = source.get("score")
                source_name = (
                    source.get("source") or source.get("document_id") or "Источник не указан"
                )
                header = f"{idx}. {source_name}"
                if isinstance(score, (int, float)):
                    header = f"{header} (score={score:.4f})"
                if text:
                    st.markdown(f"**{header}**")
                    st.caption(
                        " · ".join(
                            f"{key}: {source[key]:.4f}"
                            for key in ("rrf_score", "rerank_score")
                            if isinstance(source.get(key), (int, float))
                        )
                    )
                    st.text(text)
        else:
            st.caption("Мы пока не сформировали контекст.")

    with st.expander("Наша NLI-верификация", expanded=False):
        if claims:
            rows = []
            for claim in claims:
                claim_text = str(claim.get("claim", claim.get("text", ""))).strip()
                status = str(claim.get("status", "не указан")).strip()
                score = claim.get("score")
                line = f"**{status}** — {claim_text}".strip("— ")
                if isinstance(score, (int, float)):
                    line += f" (score={score:.3f})"
                probabilities = claim.get("probabilities") or {}
                label = (
                    max(probabilities, key=lambda key: float(probabilities[key])).removeprefix("p_")
                    if probabilities
                    else status.lower()
                )
                label = {
                    "verified": "ENTAILED",
                    "unverified": "NEUTRAL",
                    "entailment": "ENTAILED",
                    "contradiction": "CONTRADICTED",
                }.get(label, label.upper())
                icon = "🟢" if label == "ENTAILED" else "🔴" if label == "CONTRADICTED" else "🟡"
                rows.append(
                    {
                        "Наш статус NLI": f"{icon} {label}",
                        "Утверждение": claim_text,
                        "Наше решение": status,
                        "Этап": claim.get("stage", "QA"),
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("Мы не получили проверяемых утверждений.")

    with st.expander("Наши метрики", expanded=False):
        if execution_stats:
            for key, value in execution_stats.items():
                st.write(f"{key}: {value / 1000:.3f} с")
        else:
            st.caption("Мы пока не измерили время этапов.")

    policy = str(meta.get("policy", "")).strip()
    if policy:
        st.caption(policy)

    if refused:
        st.warning("Мы не можем выдать финальный ответ по политике безопасности/качества.")


def _render_chat() -> None:
    st.caption("НАШ ЛОКАЛЬНЫЙ ПОМОЩНИК")
    st.title("Разбираемся в документах вместе")

    if not st.session_state["messages"]:
        _push_message(
            {
                "role": "assistant",
                "content": "Добро пожаловать. Мы поможем вам работать с нашими документами: задайте вопрос в QA или выберите документ для резюме.",
                "mode": cast(Mode, st.session_state["mode"]),
                "timestamp": time.time(),
                "meta": None,
            }
        )

    for message in cast(list[ChatMessage], st.session_state["messages"]):
        role = cast(str, message.get("role"))
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role == "assistant":
                meta = message.get("meta")
                if meta:
                    _display_metadata(meta)


def _run_query(
    bridge: WebBridge,
    user_input: str,
    mode: Mode,
    document_id: str | None,
) -> None:
    status_messages: list[str] = []
    completed: BridgeResult | None = None

    with st.status("Мы обрабатываем ваш запрос", expanded=True) as status:
        start = time.perf_counter()

        def _on_status(stage: str) -> None:
            stage_text = stage.strip()
            status_messages.append(stage_text)
            status.write(stage_text)

        question = user_input.strip()
        if mode == "summary":
            if not question:
                question = "Подготовь структурированное резюме по текущему документу."
        else:
            question = question

        try:
            response = bridge.run(
                question=question,
                mode=mode,
                document_id=document_id,
                on_status=_on_status,
            )
            elapsed = time.perf_counter() - start
            status.write(f"Мы завершили обработку за {elapsed:.2f} с.")
            status.update(label="Мы завершили проверку", state="complete", expanded=False)

            answer = str(response.get("answer", "")).strip()
            if not answer:
                answer = "Мы не сформировали ответ. Проверим наш источник и повторим запрос."

            _push_message(
                {
                    "role": "assistant",
                    "content": answer,
                    "mode": mode,
                    "timestamp": time.perf_counter(),
                    "meta": response,
                }
            )

            completed = response
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to execute query in mode=%s: %s", mode, exc)
            status.update(label="Мы не завершили запрос", state="error", expanded=False)
            st.error("Мы не смогли получить ответ. Проверим локальный сервер и повторим запрос.")
            _push_message(
                {
                    "role": "assistant",
                    "content": "Мы не смогли выполнить запрос по внутренней ошибке. Попробуйте ещё раз, пожалуйста.",
                    "mode": mode,
                    "timestamp": time.time(),
                    "meta": {
                        "answer": "Мы не смогли выполнить запрос по внутренней ошибке. Попробуйте ещё раз, пожалуйста.",
                        "mode": mode,
                        "sources": [],
                        "claims": [],
                        "execution_stats": {},
                        "refused": False,
                        "policy": "",
                    },
                }
            )

    if completed is not None:
        with st.chat_message("assistant"):
            text = str(completed.get("answer", ""))
            st.write_stream((text[i : i + 80] for i in range(0, len(text), 80)))
            _display_metadata(completed)
    else:
        st.error("Мы не смогли получить ответ. Проверим локальный сервер и повторим запрос.")


def _ingest_file(bridge: WebBridge, file: UploadedFile) -> None:
    uploaded_bytes = file.getvalue()
    filename = str(file.name)
    status_updates: list[str] = []
    progress = st.progress(0.0)

    with st.status("Мы добавляем документ", expanded=True) as status:
        start = time.perf_counter()

        def _on_status(message: str) -> None:
            txt = message.strip()
            status_updates.append(txt)
            progress.progress(0.0, text=txt)
            status.write(txt)

        _on_status(f"Мы подготовили файл {filename}")
        if len(uploaded_bytes) > 20 * 1024 * 1024:
            status.update(label="Мы не приняли большой файл", state="error", expanded=False)
            st.error("Мы принимаем файлы до 20 МБ. Выберем документ меньшего размера.")
            return

        try:
            result = bridge.ingest(name=filename, data=uploaded_bytes, on_status=_on_status)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Upload failed for %s: %s", filename, exc)
            status.update(label="Мы не добавили документ", state="error", expanded=False)
            st.error(
                "Мы не добавили файл. Проверим формат (TXT, MD, PDF, DOCX) и доступность нашего индекса."
            )
            return

        elapsed = time.perf_counter() - start
        _on_status(f"Мы обработали файл за {elapsed:.2f} с.")
        progress.progress(1.0, text="Мы завершили добавление")
        status.update(label="Мы обработали документ", state="complete", expanded=False)
    st.success(result)


def main() -> None:
    _initialize_state()
    st.session_state["mode"] = _safe_mode(st.session_state.get("mode"))

    bridge, bridge_error = _load_bridge()
    st.session_state["bridge_error"] = bridge_error

    with st.sidebar:
        st.header("Наши документы")
        mode = st.radio(
            "Как мы работаем",
            options=("qa", "summary"),
            index=0 if st.session_state["mode"] == "qa" else 1,
            format_func=_mode_label,
            key="mode",
            on_change=_sync_placeholder,
        )
        mode = _safe_mode(st.session_state.get("mode"))
        st.caption(_mode_placeholder(mode))
        if st.session_state.get("upload_notice"):
            st.success(st.session_state.pop("upload_notice"))

        st.divider()
        st.subheader("Пополняем нашу базу")
        uploaded_file = st.file_uploader(
            "Мы принимаем TXT, Markdown, PDF и Word",
            type=["txt", "md", "pdf", "docx"],
        )
        st.caption("Мы пропускаем дубликаты. Мы извлекаем текст PDF без OCR; предел файла — 20 МБ.")
        if st.button("Добавляем в нашу базу", disabled=uploaded_file is None):
            if uploaded_file is None:
                st.info("Начнём с загрузки нашего документа.")
            elif bridge is None:
                st.error(st.session_state["bridge_error"] or "Backend недоступен.")
            else:
                _ingest_file(bridge, uploaded_file)

        st.divider()

        if mode == "summary":
            docs = _load_documents(bridge)
            if bridge is None:
                st.error(st.session_state["bridge_error"] or "Backend недоступен.")
            elif not docs:
                st.info("Мы пока не добавили документы. Начнём с загрузки файла.")
            else:
                summary_search = st.text_input(
                    "Мы ищем документ по названию",
                    key="summary_search",
                )
                query = summary_search.strip().lower()
                filtered = [
                    document
                    for document in docs
                    if query in document.name.lower() or query in document.document_id.lower()
                ]
                if not filtered and query:
                    st.info("Мы не нашли совпадений и показываем все наши документы.")
                    filtered = docs

                if filtered:
                    selected = st.selectbox(
                        "Наш документ для саммари",
                        options=[doc.document_id for doc in filtered],
                        index=0,
                        format_func=lambda doc_id: _default_doc_label(
                            next(doc for doc in filtered if doc.document_id == doc_id)
                        ),
                        key="summary_document_id",
                    )
                    st.caption(
                        f"Мы выбрали: {next(doc for doc in docs if doc.document_id == selected).name}"
                    )
                else:
                    st.warning("Мы не нашли подходящих документов.")

        else:
            st.caption("Мы ищем ответ по всей нашей базе и сохраняем ссылки на источники.")

    _render_chat()
    if mode == "summary":
        st.caption(
            "Мы обобщаем выборку фрагментов и допускаем NEUTRAL. Мы не гарантируем полноту и подтверждение всех фактов."
        )
    if mode == "summary" and st.session_state.get("summary_document_id") and bridge is not None:
        if st.button("Мы составляем саммари выбранного документа", type="primary"):
            selected_id = str(st.session_state["summary_document_id"])
            selected_name = next(
                (d.name for d in _load_documents(bridge) if d.document_id == selected_id),
                selected_id,
            )
            _push_message({"role": "user", "content": selected_name, "mode": "summary"})
            _run_query(bridge, selected_name, "summary", selected_id)

    placeholder = st.session_state["input_placeholder"]
    user_message = st.chat_input(placeholder=placeholder)
    if user_message is None:
        return

    mode = _safe_mode(st.session_state.get("mode"))
    query = user_message.strip()
    if not query:
        st.info("Начнём с вопроса или названия нашего документа.")
        return

    if bridge is None:
        st.error(st.session_state["bridge_error"] or "Backend недоступен.")
        return

    _push_message(
        {
            "role": "user",
            "content": query,
            "mode": mode,
            "timestamp": time.time(),
            "meta": None,
        }
    )
    with st.chat_message("user"):
        st.markdown(query)

    selected_doc_id: str | None = None
    if mode == "summary":
        matching = [
            d
            for d in _load_documents(bridge)
            if query.casefold() in {d.name.casefold(), d.document_id.casefold()}
        ]
        selected_doc_id = matching[0].document_id if len(matching) == 1 else None
        if not selected_doc_id:
            st.error(
                "Мы не определили единственный документ. Выберем его в списке и нажмём кнопку саммари."
            )
            return
    else:
        selected_doc_id = None

    _run_query(bridge, query, mode, selected_doc_id)


if __name__ == "__main__":
    main()
