from __future__ import annotations

import time

import streamlit as st

from src.config import load_settings
from src.corag_engine import CoRAGResult, CoRAGStep
from src.pipeline import RAGPipeline
from src.ui import (
    inject_custom_css,
    render_benchmark_tab,
    render_chat_history,
    render_sidebar,
    render_source_preview,
    render_statistics_tab,
    save_benchmark_entry,
    save_run_to_history,
)
from src.utils.logging import configure_logging


settings = load_settings()
logger = configure_logging()

st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")
inject_custom_css()

# ── Session state init ──
_defaults = {
    "pipeline": None,
    "last_answer": "",
    "last_sources": [],
    "chat_history": [],
    "chunk_size": settings.chunk_size,
    "chunk_overlap": settings.chunk_overlap,
    "chunk_benchmark": [],
    "last_question": "",
    "last_confidence": 0.0,
    "last_rationale": "",
    "last_used_query": "",
    "retrieval_mode": "vector",
    "enable_conversational": True,
    "enable_rerank": False,
    "enable_self_rag": False,
    "filter_source_names": [],
    "filter_doc_types": [],
    "corag_result": None,
    "corag_result_raw": None,
    "corag_time": 0.0,
    "enable_corag_compare": False,
    "corag_max_steps": 4,
}
for key, default in _defaults.items():
    if key not in st.session_state:
        if key == "pipeline":
            st.session_state[key] = RAGPipeline(settings=settings, logger=logger)
        else:
            st.session_state[key] = default

pipeline: RAGPipeline = st.session_state["pipeline"]

# Add global CSS for compact layout
st.markdown("""
<style>
/* Reduce top padding and move content up */
.main .block-container, [data-testid="stAppViewBlockContainer"] {
    padding-top: 0rem !important;
    padding-bottom: 1rem !important;
    margin-top: -3.5rem !important;
}

/* Hide Streamlit top header */
[data-testid="stHeader"] {
    display: none !important;
}

/* Compact tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    padding-top: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
}

/* Reduce spacing between elements */
.element-container {
    margin-bottom: 0.5rem !important;
}

/* Compact metrics */
[data-testid="stMetricValue"] {
    font-size: 1.2rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
}

/* Compact dividers */
hr {
    margin: 0.5rem 0 !important;
}

/* Compact markdown headers */
h5 {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    font-size: 1rem !important;
}

h4 {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    font-size: 1.1rem !important;
}

/* Compact text inputs */
.stTextInput, .stTextArea, .stSelectbox, .stMultiSelect {
    margin-bottom: 0.5rem !important;
}

/* Compact columns */
[data-testid="column"] {
    padding: 0.2rem !important;
}

/* Compact buttons */
.stButton > button {
    padding: 0.3rem 0.6rem;
    font-size: 0.85rem;
    min-height: 2rem;
}

/* Compact dataframes */
.stDataFrame {
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
render_sidebar(
    settings=settings,
    pipeline_ready=pipeline.is_ready,
    indexed_chunks=pipeline.document_count,
)
render_chat_history(st.session_state["chat_history"])

available_filters = pipeline.list_available_sources()

with st.sidebar:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ History"):
            st.session_state["open_clear_history_dialog"] = True
    with col_b:
        if st.button("🗑️ Vectors"):
            st.session_state["open_clear_vector_dialog"] = True


# ── Dialogs ──
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


@st.dialog("Xác nhận xóa thống kê")
def clear_stats_dialog() -> None:
    st.write("Bạn có chắc muốn xóa toàn bộ thống kê?")
    if st.button("Xóa thống kê", type="primary"):
        from pathlib import Path
        p = Path("data/run_history.json")
        if p.exists():
            p.write_text("[]", encoding="utf-8")
        st.session_state["open_clear_stats_dialog"] = False
        st.rerun()
    if st.button("Hủy", key="cancel_stats"):
        st.session_state["open_clear_stats_dialog"] = False
        st.rerun()


@st.dialog("Xác nhận xóa tài liệu")
def delete_doc_dialog() -> None:
    doc_name = st.session_state.get("delete_doc_name", "")
    st.write(f"Bạn có chắc muốn xóa tài liệu **{doc_name}**?")
    st.warning("⚠️ Hành động này không thể hoàn tác!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Xác nhận xóa", type="primary", use_container_width=True):
            removed = pipeline.remove_source(doc_name)
            if doc_name in st.session_state.get("active_docs", {}):
                del st.session_state["active_docs"][doc_name]
            st.session_state["open_delete_doc_dialog"] = False
            st.success(f"✅ Đã xóa {doc_name} ({removed} chunks)")
            st.rerun()
    with col2:
        if st.button("❌ Hủy", use_container_width=True):
            st.session_state["open_delete_doc_dialog"] = False
            st.rerun()


@st.dialog("Xác nhận xóa dòng thống kê")
def delete_stat_dialog() -> None:
    stat_index = st.session_state.get("delete_stat_index", -1)
    st.write(f"Bạn có chắc muốn xóa dòng thống kê **#{stat_index}**?")
    st.warning("⚠️ Hành động này không thể hoàn tác!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Xác nhận xóa", type="primary", use_container_width=True, key="confirm_del_stat"):
            from pathlib import Path
            import json
            
            p = Path("data/run_history.json")
            if p.exists():
                try:
                    history = json.loads(p.read_text(encoding="utf-8"))
                    # Remove entry with matching #
                    history = [h for h in history if h.get("#") != stat_index]
                    # Renumber
                    for idx, h in enumerate(history, start=1):
                        h["#"] = idx
                    p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.success(f"✅ Đã xóa dòng #{stat_index}")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
            
            st.session_state["open_delete_stat_dialog"] = False
            st.rerun()
    with col2:
        if st.button("❌ Hủy", use_container_width=True, key="cancel_del_stat"):
            st.session_state["open_delete_stat_dialog"] = False
            st.rerun()


if st.session_state.get("open_clear_history_dialog"):
    clear_history_dialog()
if st.session_state.get("open_clear_vector_dialog"):
    clear_vector_dialog()
if st.session_state.get("open_delete_doc_dialog"):
    delete_doc_dialog()
if st.session_state.get("open_delete_stat_dialog"):
    delete_stat_dialog()

# ════════════════════════════════════════════════════════
# ── MAIN TABS ──
# ════════════════════════════════════════════════════════
tab_docs, tab_rag, tab_compare, tab_stats, tab_bench = st.tabs(
    ["📚 Tài liệu", "🔍 RAG", "⚔️ RAG vs CoRAG", "📊 Thống kê", "📐 Benchmark"]
)

# ── TAB 0: Documents ──
with tab_docs:
    st.markdown("#### 📚 Quản lý Tài liệu")
    
    # Upload section
    uploaded_files = st.file_uploader(
        "Chọn file PDF hoặc DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="docs_uploader",
        help="Kéo thả file vào đây hoặc click để chọn. Giới hạn 200MB/file"
    )
    
    # Save to session state for other tabs to access
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        st.success(f"✅ {len(uploaded_files)} file")
    
    # Process documents section
    st.markdown("**⚡ Xử lý tài liệu**")
    
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        process_chunk_size = st.selectbox(
            "Chunk Size",
            options=[500, 1000, 1500, 2000],
            index=[500, 1000, 1500, 2000].index(st.session_state["chunk_size"])
            if st.session_state["chunk_size"] in [500, 1000, 1500, 2000]
            else 1,
            key="process_chunk_size",
        )
    with pcol2:
        process_chunk_overlap = st.selectbox(
            "Chunk Overlap",
            options=[50, 100, 200],
            index=[50, 100, 200].index(st.session_state["chunk_overlap"])
            if st.session_state["chunk_overlap"] in [50, 100, 200]
            else 1,
            key="process_chunk_overlap",
        )
    
    process_clicked = st.button("⚡ Process Document(s)", type="primary", disabled=not uploaded_files)
    if process_clicked and uploaded_files:
        try:
            with st.spinner("Đang xử lý tài liệu..."):
                result = pipeline.ingest_files(
                    uploaded_files,
                    chunk_size=process_chunk_size,
                    chunk_overlap=process_chunk_overlap,
                )
            st.success(
                f"✅ {result.files_indexed} file(s), {result.chunks} chunks trong {result.seconds:.2f}s "
                f"(size={result.chunk_size}, overlap={result.chunk_overlap})"
            )
            st.rerun()
        except Exception as exc:
            logger.exception("Document processing failed")
            st.error(f"Lỗi: {exc}")
    
    # Document Manager
    st.markdown("**📋 Tài liệu đã index**")
    
    indexed_docs = pipeline.list_indexed_documents()
    available_filters = pipeline.list_available_sources()
    
    # Filters
    if available_filters.get("source_name") or available_filters.get("doc_type"):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.session_state["filter_source_names"] = st.multiselect(
                "Nguồn (Source)",
                options=available_filters.get("source_name", []),
                default=[i for i in st.session_state.get("filter_source_names", []) if i in available_filters.get("source_name", [])],
                key="docs_filter_sources"
            )
        with fcol2:
            st.session_state["filter_doc_types"] = st.multiselect(
                "Loại file (Type)",
                options=available_filters.get("doc_type", []),
                default=[i for i in st.session_state.get("filter_doc_types", []) if i in available_filters.get("doc_type", [])],
                key="docs_filter_types"
            )
    
    doc_count = len(indexed_docs)
    total_chunks = sum(d.get("Số chunks", 0) for d in indexed_docs)
    st.markdown(
        f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.5rem;margin-top:1rem;">'
        f'<span style="font-weight:600;color:#2563EB;">📚 {doc_count} tài liệu đã index</span>'
        f'<span style="color:#64748B;font-size:0.8rem;margin-left:1rem;">({total_chunks} chunks)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    if "active_docs" not in st.session_state:
        # By default, use all documents
        st.session_state["active_docs"] = {d.get("Tên file"): True for d in indexed_docs}
    
    # Display documents as table
    if indexed_docs:
        # Create columns for table with actions
        cols = st.columns([0.5, 3, 1, 1, 1.5, 1, 1])
        
        # Header row
        cols[0].markdown("**Icon**")
        cols[1].markdown("**Tên file**")
        cols[2].markdown("**Loại**")
        cols[3].markdown("**Chunks**")
        cols[4].markdown("**Ngày upload**")
        cols[5].markdown("**Sử dụng**")
        cols[6].markdown("**Xóa**")
        
        st.divider()
        
        # Data rows
        for i, doc_info in enumerate(indexed_docs):
            fname = doc_info.get("Tên file", "?")
            dtype = doc_info.get("Loại", "?")
            chunks = doc_info.get("Số chunks", 0)
            up_time = doc_info.get("Ngày upload", "?")
            icon = "📕" if dtype == "PDF" else "📘" if dtype == "DOCX" else "📄"
            
            # Auto-select new docs
            if fname not in st.session_state["active_docs"]:
                st.session_state["active_docs"][fname] = True
            
            # Create row
            row_cols = st.columns([0.5, 3, 1, 1, 1.5, 1, 1])
            
            row_cols[0].markdown(icon)
            row_cols[1].markdown(f"{fname}")
            row_cols[2].markdown(dtype)
            row_cols[3].markdown(str(chunks))
            row_cols[4].markdown(up_time)
            
            # Action: Use checkbox
            is_active = row_cols[5].checkbox(
                "✓",
                value=st.session_state["active_docs"].get(fname, True),
                key=f"docs_use_{i}",
                label_visibility="collapsed"
            )
            st.session_state["active_docs"][fname] = is_active
            
            # Action: Delete button
            if row_cols[6].button("🗑️", key=f"docs_del_{i}", help=f"Xóa {fname}"):
                st.session_state["open_delete_doc_dialog"] = True
                st.session_state["delete_doc_name"] = fname
                st.session_state["delete_doc_index"] = i
                st.rerun()
            
            if i < len(indexed_docs) - 1:
                st.divider()
    else:
        st.info("📭 Chưa có tài liệu nào được index. Upload và process tài liệu ở trên.")

# ── TAB 1: RAG ──
with tab_rag:
    # ── Inline config row ──
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        st.session_state["chunk_size"] = st.selectbox(
            "⚙️ Chunk Size",
            options=[500, 1000, 1500, 2000],
            index=[500, 1000, 1500, 2000].index(st.session_state["chunk_size"])
            if st.session_state["chunk_size"] in [500, 1000, 1500, 2000]
            else 1,
            key="rag_chunk_size",
        )
    with cfg2:
        st.session_state["chunk_overlap"] = st.selectbox(
            "⚙️ Chunk Overlap",
            options=[50, 100, 200],
            index=[50, 100, 200].index(st.session_state["chunk_overlap"])
            if st.session_state["chunk_overlap"] in [50, 100, 200]
            else 1,
            key="rag_chunk_overlap",
        )
    with cfg3:
        st.session_state["retrieval_mode"] = st.selectbox(
            "🔍 Retrieval Mode",
            options=["vector", "hybrid", "keyword"],
            index=["vector", "hybrid", "keyword"].index(st.session_state["retrieval_mode"]),
            key="rag_retrieval_mode",
        )

    # ── Feature toggles ──
    tog1, tog2, tog3 = st.columns(3)
    with tog1:
        st.session_state["enable_conversational"] = st.checkbox(
            "💬 Conversational", value=st.session_state["enable_conversational"], key="rag_conv"
        )
    with tog2:
        st.session_state["enable_rerank"] = st.checkbox(
            "🔀 Re-ranking", value=st.session_state["enable_rerank"], key="rag_rerank"
        )
    with tog3:
        st.session_state["enable_self_rag"] = st.checkbox(
            "🧠 Self-RAG", value=st.session_state["enable_self_rag"], key="rag_selfrag"
        )

    st.markdown("#### 💡 Đặt câu hỏi")
    question = st.text_input("Nhập câu hỏi về nội dung tài liệu", key="rag_question")

    ask_clicked = st.button(
        "🚀 Get Answer", disabled=not pipeline.is_ready or not question.strip(), key="rag_ask"
    )
    if ask_clicked:
        try:
            st.session_state["corag_result"] = None
            st.session_state["corag_result_raw"] = None
            st.session_state["corag_time"] = 0.0

            # Combine top filters + document manager active state
            active_docs_from_manager = [fname for fname, is_active in st.session_state.get("active_docs", {}).items() if is_active]
            selected_sources_from_filter = st.session_state.get("filter_source_names", [])
            
            # Intersection: Must be checked in the manager AND (if filter applied) in the filter list
            if selected_sources_from_filter:
                final_sources = [s for s in active_docs_from_manager if s in selected_sources_from_filter]
            else:
                final_sources = active_docs_from_manager

            metadata_filters = {
                "source_name": final_sources,
                "doc_type": st.session_state.get("filter_doc_types", []),
            }
            with st.spinner("🔄 Đang truy xuất và sinh câu trả lời..."):
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
                f"✅ [{answer_result.mode}] | retrieve={answer_result.retrieval_seconds:.2f}s, "
                f"generate={answer_result.generation_seconds:.2f}s"
            )

            # Save to statistics immediately
            source_names = list({
                doc.metadata.get("source_name", "?") for doc in answer_result.sources
            })
            save_run_to_history({
                "Câu hỏi": question.strip(),
                "Mode": answer_result.mode,
                "Conversational": "✅" if st.session_state["enable_conversational"] else "❌",
                "Rerank": "✅" if st.session_state["enable_rerank"] else "❌",
                "Self-RAG": "✅" if st.session_state["enable_self_rag"] else "❌",
                "Confidence": f"{answer_result.confidence:.0%}",
                "Retrieval (ms)": f"{answer_result.retrieval_seconds * 1000:,.0f}",
                "Generation (ms)": f"{answer_result.generation_seconds * 1000:,.0f}",
                "Tổng (ms)": f"{(answer_result.retrieval_seconds + answer_result.generation_seconds) * 1000:,.0f}",
                "Nguồn": ", ".join(source_names),
                "Số nguồn": len(answer_result.sources),
                "Query dùng": answer_result.used_query,
                "_full_question": question.strip(),
                "_full_answer": answer_result.answer,
            })



        except Exception as exc:
            logger.exception("Question answering failed")
            st.error(f"❌ Lỗi khi trả lời: {exc}")

    # Show RAG response
    if st.session_state["last_answer"]:
        st.markdown("#### 📝 Câu trả lời")
        st.write(st.session_state["last_answer"])
        st.caption(
            f"🎯 confidence={st.session_state['last_confidence']:.2f} | "
            f"🔎 query={st.session_state['last_used_query']}"
        )
        if st.session_state["last_rationale"]:
            st.caption(f"🧠 {st.session_state['last_rationale']}")

    render_source_preview(query=st.session_state["last_question"])


# ── TAB 2: RAG vs CoRAG ──
with tab_compare:
    st.markdown("#### ⚔️ So sánh RAG vs CoRAG")
    
    compare_question = st.text_input("Nhập câu hỏi để so sánh RAG và CoRAG", key="compare_question")
    compare_clicked = st.button("🚀 Chạy So Sánh", disabled=not pipeline.is_ready or not compare_question.strip(), key="compare_btn")
    
    if compare_clicked:
        try:
            st.session_state["compare_rag_result"] = None
            st.session_state["compare_corag_result"] = None
            
            # Combine top filters + document manager active state
            active_docs_from_manager = [fname for fname, is_active in st.session_state.get("active_docs", {}).items() if is_active]
            selected_sources_from_filter = st.session_state.get("filter_source_names", [])
            
            if selected_sources_from_filter:
                final_sources = [s for s in active_docs_from_manager if s in selected_sources_from_filter]
            else:
                final_sources = active_docs_from_manager

            metadata_filters = {
                "source_name": final_sources,
                "doc_type": st.session_state.get("filter_doc_types", []),
            }

            col_run_rag, col_run_corag = st.columns(2)
            
            # Run RAG
            with col_run_rag:
                rag_t0 = time.perf_counter()
                with st.status("🔍 RAG đang xử lý...", expanded=True) as rag_status:
                    rag_result = pipeline.answer(
                        compare_question.strip(),
                        chat_history=[],
                        retrieval_mode=st.session_state.get("retrieval_mode", "vector"),
                        metadata_filters=metadata_filters,
                        enable_rerank=st.session_state.get("enable_rerank", False),
                        enable_self_rag=st.session_state.get("enable_self_rag", False),
                        conversational=False,
                    )
                    rag_time = time.perf_counter() - rag_t0
                    rag_status.update(label=f"✅ RAG hoàn thành ({rag_time:.2f}s)!", state="complete")
                    st.session_state["compare_rag_result"] = rag_result
                    st.session_state["compare_rag_time"] = rag_time

            # Run CoRAG
            with col_run_corag:
                corag_t0 = time.perf_counter()
                with st.status("⚡ CoRAG đang suy luận...", expanded=True) as corag_status:
                    corag_steps_placeholder = st.empty()
                    live_steps: list[CoRAGStep] = []

                    def on_step(step: CoRAGStep) -> None:
                        live_steps.append(step)
                        with corag_steps_placeholder.container():
                            for item in live_steps:
                                st.caption(
                                    f"Step {item.step}: '{item.query[:50]}...' "
                                    f"-> {item.retrieved_count} docs, sufficient={item.sufficient}"
                                )

                    corag_result = pipeline.answer_corag(
                        question=compare_question.strip(),
                        max_steps=st.session_state.get("corag_max_steps", 4),
                        retrieval_mode=st.session_state.get("retrieval_mode", "vector"),
                        metadata_filters=metadata_filters,
                        enable_rerank=st.session_state.get("enable_rerank", False),
                        step_callback=on_step,
                    )
                    corag_time = time.perf_counter() - corag_t0
                    st.session_state["compare_corag_time"] = corag_time
                    st.session_state["compare_corag_result"] = corag_result
                    st.session_state["compare_corag_result_raw"] = pipeline._last_corag_result
                    corag_status.update(label=f"✅ CoRAG hoàn thành ({corag_time:.2f}s)!", state="complete")
        except Exception as exc:
            logger.exception("Comparison failed")
            st.error(f"❌ Lỗi khi so sánh: {exc}")

    # Display comparison if available
    if st.session_state.get("compare_rag_result") and st.session_state.get("compare_corag_result"):
        rag_res = st.session_state["compare_rag_result"]
        corag_res = st.session_state["compare_corag_result"]
        raw_res = st.session_state.get("compare_corag_result_raw")

        col_rag, col_corag = st.columns(2)
        with col_rag:
            st.markdown(
                '<div class="card" style="border-left:3px solid #2563EB;">'
                f'<div class="card-header">🔍 RAG ({st.session_state.get("compare_rag_time", 0):.2f}s)</div>'
                f'<div class="card-body">{rag_res.answer}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with col_corag:
            st.markdown(
                '<div class="card" style="border-left:3px solid #10B981;">'
                f'<div class="card-header">⚡ CoRAG ({st.session_state.get("compare_corag_time", 0):.2f}s)</div>'
                f'<div class="card-body">{corag_res.answer}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("🔢 Steps", f"RAG: 1 / CoRAG: {raw_res.steps if isinstance(raw_res, CoRAGResult) else 'N/A'}")
        m2.metric(
            "📄 Docs",
            f"RAG: {len(rag_res.sources)} / CoRAG: {raw_res.total_docs if isinstance(raw_res, CoRAGResult) else len(corag_res.sources)}",
        )
        m3.metric(
            "🎯 Confidence",
            f"RAG: {rag_res.confidence:.2f} / CoRAG: {corag_res.confidence:.2f}",
        )

        if isinstance(raw_res, CoRAGResult):
            st.markdown("#### 🔗 CoRAG Chain Trace")
            for step in raw_res.chain:
                label = f"Step {step.step}: {step.query[:60]}{'...' if len(step.query) > 60 else ''}"
                with st.expander(label, expanded=False):
                    st.write(f"📥 Retrieved: {step.retrieved_count} new docs")
                    st.write(f"✅ Sufficient: {step.sufficient}")
                    if step.discovered_entities:
                        st.write(f"🏷️ Entities: {step.discovered_entities}")
                    if step.missing_parts:
                        st.write(f"❓ Missing: {', '.join(step.missing_parts)}")
                    if step.next_query:
                        st.write(f"➡️ Next: {step.next_query}")
                    st.caption(f"💭 {step.reasoning}")


# ── TAB 3: Statistics ──
with tab_stats:
    render_statistics_tab()


# ── TAB 4: Benchmark ──
with tab_bench:
    render_benchmark_tab()

    # Handle benchmark execution with progressive display
    uploaded_files = st.session_state.get("uploaded_files", [])
    if st.session_state.get("bm_should_run") and uploaded_files:
        st.session_state["bm_should_run"] = False
        configs = st.session_state.get("bm_configs", [])
        bm_q = st.session_state.get("bm_q", "")
        bm_ref_answer = st.session_state.get("bm_ref_answer", "")
        bm_session_id = st.session_state.get("bm_session_id", None)

        if configs and bm_q:
            results_container = st.container()
            progress_bar = st.progress(0, text="Đang chạy benchmark...")
            results_placeholder = st.empty()
            completed_results: list[dict] = []

            for i, (cs, co) in enumerate(configs):
                progress_bar.progress(
                    (i) / len(configs),
                    text=f"⏳ Đang chạy config {i+1}/{len(configs)}: size={cs}, overlap={co}..."
                )
                try:
                    result = pipeline.benchmark_single_config(
                        uploaded_files=uploaded_files,
                        question=bm_q,
                        chunk_size=cs,
                        chunk_overlap=co,
                        retrieval_mode=st.session_state["retrieval_mode"],
                    )
                    
                    # Calculate metrics if reference answer is provided
                    if bm_ref_answer:
                        from src.ui import calculate_em, calculate_cem, calculate_f1
                        result["reference_answer"] = bm_ref_answer
                        result["EM"] = f"{calculate_em(result['answer'], bm_ref_answer):.2f}"
                        result["CEM"] = f"{calculate_cem(result['answer'], bm_ref_answer):.2f}"
                        result["F1"] = f"{calculate_f1(result['answer'], bm_ref_answer):.2f}"
                    
                    # Save immediately with session ID
                    save_benchmark_entry(result, benchmark_session_id=bm_session_id)
                    completed_results.append(result)

                    # Show progressive results
                    with results_placeholder.container():
                        st.markdown(f"##### ✅ Hoàn thành {len(completed_results)}/{len(configs)}")
                        if bm_session_id:
                            st.caption(f"📁 Session: `{bm_session_id}`")
                        
                        display_rows = []
                        for r in completed_results:
                            row = {
                                "Size": r["chunk_size"],
                                "Overlap": r["chunk_overlap"],
                                "Chunks": r["total_chunks"],
                                "Retrieval (ms)": r["retrieval_ms"],
                                "Generation (ms)": r["generation_ms"],
                                "Total (ms)": r["total_ms"],
                            }
                            if bm_ref_answer:
                                row["EM"] = r.get("EM", "—")
                                row["CEM"] = r.get("CEM", "—")
                                row["F1"] = r.get("F1", "—")
                            display_rows.append(row)
                        st.dataframe(display_rows, use_container_width=True, hide_index=True)

                        # Show latest answer
                        metrics_info = ""
                        if bm_ref_answer:
                            metrics_info = f" | EM={result.get('EM', '—')} CEM={result.get('CEM', '—')} F1={result.get('F1', '—')}"
                        
                        st.markdown(
                            f'<div class="card" style="border-left:3px solid #10B981;">'
                            f'<div class="card-header">✅ size={cs} overlap={co} — {result["total_ms"]}ms{metrics_info}</div>'
                            f'<div class="card-body">{result["answer"][:300]}{"..." if len(result["answer"]) > 300 else ""}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as exc:
                    logger.exception(f"Benchmark config size={cs} overlap={co} failed")
                    with results_placeholder.container():
                        st.error(f"❌ size={cs} overlap={co}: {exc}")

            progress_bar.progress(1.0, text=f"✅ Benchmark hoàn tất! {len(completed_results)} configs")
            st.balloons()

    elif st.session_state.get("bm_should_run") and not uploaded_files:
        st.session_state["bm_should_run"] = False
        st.error("📁 Vui lòng upload tài liệu trước khi chạy benchmark.")


# ── Clear benchmark dialog ──
@st.dialog("Xác nhận xóa lịch sử benchmark")
def clear_bm_dialog() -> None:
    st.write("Bạn có chắc muốn xóa toàn bộ lịch sử benchmark?")
    if st.button("Xóa benchmark", type="primary"):
        from pathlib import Path
        p = Path("data/benchmark_history.json")
        if p.exists():
            p.write_text("[]", encoding="utf-8")
        st.session_state["open_clear_bm_dialog"] = False
        st.rerun()
    if st.button("Hủy", key="cancel_bm"):
        st.session_state["open_clear_bm_dialog"] = False
        st.rerun()

if st.session_state.get("open_clear_bm_dialog"):
    clear_bm_dialog()
