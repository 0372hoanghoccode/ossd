from __future__ import annotations

import streamlit as st

from src.config import load_settings
from src.pipeline import RAGPipeline
from src.ui import render_chat_history, render_sidebar, render_source_preview
from src.utils.logging import configure_logging


settings = load_settings()
logger = configure_logging()

st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = RAGPipeline(settings=settings, logger=logger)
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "chunk_size" not in st.session_state:
    st.session_state["chunk_size"] = settings.chunk_size
if "chunk_overlap" not in st.session_state:
    st.session_state["chunk_overlap"] = settings.chunk_overlap
if "chunk_benchmark" not in st.session_state:
    st.session_state["chunk_benchmark"] = []
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "last_confidence" not in st.session_state:
    st.session_state["last_confidence"] = 0.0
if "last_rationale" not in st.session_state:
    st.session_state["last_rationale"] = ""
if "last_used_query" not in st.session_state:
    st.session_state["last_used_query"] = ""
if "retrieval_mode" not in st.session_state:
    st.session_state["retrieval_mode"] = "vector"
if "enable_conversational" not in st.session_state:
    st.session_state["enable_conversational"] = True
if "enable_rerank" not in st.session_state:
    st.session_state["enable_rerank"] = False
if "enable_self_rag" not in st.session_state:
    st.session_state["enable_self_rag"] = False
if "filter_source_names" not in st.session_state:
    st.session_state["filter_source_names"] = []
if "filter_doc_types" not in st.session_state:
    st.session_state["filter_doc_types"] = []

pipeline: RAGPipeline = st.session_state["pipeline"]

st.title("SmartDoc AI - Intelligent Document Q&A")
st.caption("Upload nhiều PDF/DOCX và hỏi đáp tiếng Việt/tiếng Anh với advanced RAG.")

uploaded_files = st.file_uploader("Upload tài liệu", type=["pdf", "docx"], accept_multiple_files=True)

render_sidebar(
    settings=settings,
    pipeline_ready=pipeline.is_ready,
    indexed_chunks=pipeline.document_count,
)
render_chat_history(st.session_state["chat_history"])

available_filters = pipeline.list_available_sources()

with st.sidebar:
    st.subheader("Chunk Strategy")
    st.session_state["chunk_size"] = st.selectbox(
        "chunk_size",
        options=[500, 1000, 1500, 2000],
        index=[500, 1000, 1500, 2000].index(st.session_state["chunk_size"])
        if st.session_state["chunk_size"] in [500, 1000, 1500, 2000]
        else 1,
    )
    st.session_state["chunk_overlap"] = st.selectbox(
        "chunk_overlap",
        options=[50, 100, 200],
        index=[50, 100, 200].index(st.session_state["chunk_overlap"])
        if st.session_state["chunk_overlap"] in [50, 100, 200]
        else 1,
    )

    st.subheader("Advanced Retrieval")
    st.session_state["retrieval_mode"] = st.selectbox(
        "retrieval_mode",
        options=["vector", "hybrid", "keyword"],
        index=["vector", "hybrid", "keyword"].index(st.session_state["retrieval_mode"]),
    )
    st.session_state["enable_conversational"] = st.checkbox(
        "Conversational RAG",
        value=st.session_state["enable_conversational"],
    )
    st.session_state["enable_rerank"] = st.checkbox(
        "Re-ranking (Cross-Encoder)",
        value=st.session_state["enable_rerank"],
    )
    st.session_state["enable_self_rag"] = st.checkbox(
        "Self-RAG reflection",
        value=st.session_state["enable_self_rag"],
    )

    st.subheader("Metadata Filter")
    st.session_state["filter_source_names"] = st.multiselect(
        "source_name",
        options=available_filters.get("source_name", []),
        default=[item for item in st.session_state["filter_source_names"] if item in available_filters.get("source_name", [])],
    )
    st.session_state["filter_doc_types"] = st.multiselect(
        "doc_type",
        options=available_filters.get("doc_type", []),
        default=[item for item in st.session_state["filter_doc_types"] if item in available_filters.get("doc_type", [])],
    )

    run_benchmark_clicked = st.button("Run Chunk Benchmark")

    if st.button("Clear History"):
        st.session_state["open_clear_history_dialog"] = True
    if st.button("Clear Vector Store"):
        st.session_state["open_clear_vector_dialog"] = True


@st.dialog("Xác nhận xóa lịch sử")
def clear_history_dialog() -> None:
    st.write("Bạn có chắc muốn xóa toàn bộ lịch sử hội thoại?")
    if st.button("Xóa lịch sử", type="primary"):
        st.session_state["chat_history"] = []
        st.session_state["last_answer"] = ""
        st.session_state["last_sources"] = []
        st.session_state["open_clear_history_dialog"] = False
        st.rerun()
    if st.button("Hủy"):
        st.session_state["open_clear_history_dialog"] = False
        st.rerun()


@st.dialog("Xác nhận xóa vector store")
def clear_vector_dialog() -> None:
    st.write("Bạn có chắc muốn xóa tài liệu đã index và vector store?")
    if st.button("Xóa vector store", type="primary"):
        pipeline.clear_vector_store()
        st.session_state["last_answer"] = ""
        st.session_state["last_sources"] = []
        st.session_state["open_clear_vector_dialog"] = False
        st.rerun()
    if st.button("Hủy", key="cancel_vector"):
        st.session_state["open_clear_vector_dialog"] = False
        st.rerun()


if st.session_state.get("open_clear_history_dialog"):
    clear_history_dialog()
if st.session_state.get("open_clear_vector_dialog"):
    clear_vector_dialog()

if run_benchmark_clicked:
    if not uploaded_files:
        st.error("Vui lòng upload tài liệu trước khi chạy benchmark.")
    else:
        try:
            with st.spinner("Đang benchmark các cấu hình chunk..."):
                benchmark_results = pipeline.benchmark_chunk_strategies(
                    uploaded_file=uploaded_files[0],
                    chunk_sizes=[500, 1000, 1500, 2000],
                    chunk_overlaps=[50, 100, 200],
                )
            st.session_state["chunk_benchmark"] = [
                {
                    "chunk_size": item.chunk_size,
                    "chunk_overlap": item.chunk_overlap,
                    "chunks": item.chunks,
                    "split_seconds": round(item.split_seconds, 4),
                    "accuracy_recall_at_k": round(item.accuracy_recall_at_k, 4),
                    "benchmark_queries": item.benchmark_queries,
                }
                for item in benchmark_results
            ]
            st.success("Benchmark hoàn tất")
        except Exception as exc:
            logger.exception("Chunk benchmark failed")
            st.error(f"Lỗi benchmark chunk strategy: {exc}")

process_clicked = st.button("Process Document(s)", type="primary", disabled=not uploaded_files)
if process_clicked and uploaded_files:
    try:
        with st.spinner("Đang xử lý tài liệu..."):
            result = pipeline.ingest_files(
                uploaded_files,
                chunk_size=st.session_state["chunk_size"],
                chunk_overlap=st.session_state["chunk_overlap"],
            )
        st.success(
            f"Đã xử lý xong: {result.files_indexed} file(s), {result.chunks} chunks trong {result.seconds:.2f}s "
            f"(chunk_size={result.chunk_size}, overlap={result.chunk_overlap})"
        )
    except Exception as exc:
        logger.exception("Document processing failed")
        st.error(f"Lỗi xử lý tài liệu: {exc}")

if st.session_state["chunk_benchmark"]:
    st.subheader("Chunk Benchmark Results")
    st.dataframe(st.session_state["chunk_benchmark"], use_container_width=True)

st.divider()
st.subheader("Ask a Question")
question = st.text_input("Nhập câu hỏi về nội dung tài liệu")

ask_clicked = st.button("Get Answer", disabled=not pipeline.is_ready or not question.strip())
if ask_clicked:
    try:
        metadata_filters = {
            "source_name": st.session_state["filter_source_names"],
            "doc_type": st.session_state["filter_doc_types"],
        }
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            answer_result = pipeline.answer(
                question.strip(),
                chat_history=st.session_state["chat_history"],
                retrieval_mode=st.session_state["retrieval_mode"],
                metadata_filters=metadata_filters,
                enable_rerank=st.session_state["enable_rerank"],
                enable_self_rag=st.session_state["enable_self_rag"],
                conversational=st.session_state["enable_conversational"],
            )
        st.session_state["last_answer"] = answer_result.answer
        st.session_state["last_sources"] = answer_result.sources
        st.session_state["last_question"] = question.strip()
        st.session_state["last_confidence"] = answer_result.confidence
        st.session_state["last_rationale"] = answer_result.rationale
        st.session_state["last_used_query"] = answer_result.used_query
        st.session_state["chat_history"].append(
            {"question": question.strip(), "answer": answer_result.answer}
        )
        st.success(
            f"Done [{answer_result.mode}] | retrieve={answer_result.retrieval_seconds:.2f}s, "
            f"generate={answer_result.generation_seconds:.2f}s"
        )
    except Exception as exc:
        logger.exception("Question answering failed")
        st.error(f"Lỗi khi trả lời: {exc}")

if st.session_state["last_answer"]:
    st.subheader("Response")
    st.write(st.session_state["last_answer"])
    st.caption(
        f"confidence={st.session_state['last_confidence']:.2f} | used_query={st.session_state['last_used_query']}"
    )
    if st.session_state["last_rationale"]:
        st.caption(f"self-rag rationale: {st.session_state['last_rationale']}")

render_source_preview(query=st.session_state["last_question"])
