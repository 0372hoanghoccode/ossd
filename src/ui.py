from __future__ import annotations

import html
import re

import streamlit as st

from src.config import Settings


def render_sidebar(settings: Settings, pipeline_ready: bool, indexed_chunks: int = 0) -> None:
    with st.sidebar:
        st.header("SmartDoc AI")
        st.write("Hệ thống hỏi đáp tài liệu bằng RAG chạy local.")

        st.subheader("Model")
        st.write(f"LLM: {settings.ollama_model}")
        st.write(f"Embedding: {settings.embedding_model}")

        st.subheader("Retriever")
        st.write(f"search_type: {settings.retriever_search_type}")
        st.write(f"k: {settings.retriever_k}")

        st.subheader("Chunking")
        st.write(f"chunk_size: {settings.chunk_size}")
        st.write(f"chunk_overlap: {settings.chunk_overlap}")

        st.divider()
        st.caption("Ready" if pipeline_ready else "Upload PDF để bắt đầu")
        st.caption(f"Indexed chunks: {indexed_chunks}")


def _highlight_query_terms(text: str, query: str) -> str:
    escaped_text = html.escape(text)
    terms = [term.strip() for term in re.split(r"\s+", query) if len(term.strip()) >= 3]
    if not terms:
        return escaped_text

    pattern = re.compile("(" + "|".join(re.escape(term) for term in terms[:8]) + ")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped_text)


def render_source_preview(query: str = "") -> None:
    if "last_sources" not in st.session_state or not st.session_state["last_sources"]:
        return

    st.subheader("Citations")
    for idx, doc in enumerate(st.session_state["last_sources"], start=1):
        source_name = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        start_index = doc.metadata.get("start_index", "?")
        st.markdown(
            f"[{idx}] **{source_name}** | page: {page} | chunk: {chunk_id} | start: {start_index}"
        )

    with st.expander("Nguồn tham chiếu (top chunks)"):
        options = []
        for idx, doc in enumerate(st.session_state["last_sources"], start=1):
            source_name = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            options.append(f"#{idx} | {source_name} | page {page} | chunk {chunk_id}")

        selected = st.selectbox("Chọn đoạn để xem context gốc", options=options)
        selected_index = options.index(selected)
        selected_doc = st.session_state["last_sources"][selected_index]
        highlighted = _highlight_query_terms(selected_doc.page_content, query)
        st.markdown(highlighted, unsafe_allow_html=True)

        st.divider()
        st.caption("Danh sách top chunks")
        for idx, doc in enumerate(st.session_state["last_sources"], start=1):
            source = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            st.markdown(f"**#{idx}** | source: {source} | page: {page} | chunk: {chunk_id}")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))


def render_chat_history(history: list[dict[str, str]]) -> None:
    with st.sidebar:
        st.subheader("Chat History")
        if not history:
            st.caption("Chưa có lịch sử hội thoại")
            return

        for idx, item in enumerate(reversed(history), start=1):
            st.markdown(f"**#{idx} Q:** {item['question']}")
            st.caption(f"A: {item['answer'][:120]}{'...' if len(item['answer']) > 120 else ''}")
