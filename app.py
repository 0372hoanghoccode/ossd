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

# ── Header ──
st.markdown(
    "<div style='text-align:center;margin-bottom:0.5rem;'>"
    "<h1 style='color:#2563EB;margin:0;font-size:1.6rem;'>📄 SmartDoc AI</h1>"
    "<p style='color:#64748B;font-size:0.85rem;margin:0;'>Upload PDF/DOCX · Hỏi đáp thông minh · RAG & CoRAG</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Upload ──
uploaded_files = st.file_uploader(
    "📁 Upload tài liệu", type=["pdf", "docx"], accept_multiple_files=True
)

# ── Document Manager (persisted) ──
indexed_docs = pipeline.list_indexed_documents()
# ── Move filters here ──
available_filters = pipeline.list_available_sources()
if available_filters.get("source_name") or available_filters.get("doc_type"):
    st.markdown("##### 🏷️ Lọc Tài Liệu")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.session_state["filter_source_names"] = st.multiselect(
            "Nguồn (Source)",
            options=available_filters.get("source_name", []),
            default=[i for i in st.session_state.get("filter_source_names", []) if i in available_filters.get("source_name", [])],
            key="top_filter_sources"
        )
    with fcol2:
        st.session_state["filter_doc_types"] = st.multiselect(
            "Loại file (Type)",
            options=available_filters.get("doc_type", []),
            default=[i for i in st.session_state.get("filter_doc_types", []) if i in available_filters.get("doc_type", [])],
            key="top_filter_types"
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
            key=f"use_doc_{i}",
            label_visibility="collapsed"
        )
        st.session_state["active_docs"][fname] = is_active
        
        # Action: Delete button
        if row_cols[6].button("🗑️", key=f"del_doc_{i}", help=f"Xóa {fname}"):
            st.session_state["open_delete_doc_dialog"] = True
            st.session_state["delete_doc_name"] = fname
            st.session_state["delete_doc_index"] = i
            st.rerun()
        
        if i < len(indexed_docs) - 1:
            st.divider()

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


if st.session_state.get("open_clear_history_dialog"):
    clear_history_dialog()
if st.session_state.get("open_clear_vector_dialog"):
    clear_vector_dialog()
if st.session_state.get("open_clear_stats_dialog"):
    clear_stats_dialog()
if st.session_state.get("open_delete_doc_dialog"):
    delete_doc_dialog()

# ── Process documents ──
process_clicked = st.button("⚡ Process Document(s)", type="primary", disabled=not uploaded_files)
if process_clicked and uploaded_files:
    try:
        with st.spinner("Đang xử lý tài liệu..."):
            result = pipeline.ingest_files(
                uploaded_files,
                chunk_size=st.session_state["chunk_size"],
                chunk_overlap=st.session_state["chunk_overlap"],
            )
        st.success(
            f"✅ {result.files_indexed} file(s), {result.chunks} chunks trong {result.seconds:.2f}s "
            f"(size={result.chunk_size}, overlap={result.chunk_overlap})"
        )
    except Exception as exc:
        logger.exception("Document processing failed")
        st.error(f"Lỗi xử lý tài liệu: {exc}")

# ════════════════════════════════════════════════════════
# ── MAIN TABS ──
# ════════════════════════════════════════════════════════
st.divider()
tab_rag, tab_compare, tab_stats, tab_bench = st.tabs(
    ["🔍 RAG", "⚔️ RAG vs CoRAG", "📊 Thống kê", "📐 Benchmark"]
)

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

            # CoRAG comparison
            if st.session_state.get("enable_corag_compare"):
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

                    corag_t0 = time.perf_counter()
                    corag_result = pipeline.answer_corag(
                        question=question.strip(),
                        max_steps=st.session_state.get("corag_max_steps", 4),
                        retrieval_mode=st.session_state["retrieval_mode"],
                        metadata_filters=metadata_filters,
                        enable_rerank=st.session_state["enable_rerank"],
                        step_callback=on_step,
                    )
                    st.session_state["corag_time"] = time.perf_counter() - corag_t0
                    st.session_state["corag_result"] = corag_result
                    st.session_state["corag_result_raw"] = pipeline._last_corag_result
                    corag_status.update(label="✅ CoRAG hoàn thành!", state="complete")

                    # Save CoRAG to statistics too
                    save_run_to_history({
                        "Câu hỏi": question.strip(),
                        "Mode": corag_result.mode,
                        "Conversational": "❌",
                        "Rerank": "✅" if st.session_state["enable_rerank"] else "❌",
                        "Self-RAG": "❌",
                        "Confidence": f"{corag_result.confidence:.0%}",
                        "Retrieval (ms)": f"{corag_result.retrieval_seconds * 1000:,.0f}",
                        "Generation (ms)": "—",
                        "Tổng (ms)": f"{st.session_state['corag_time'] * 1000:,.0f}",
                        "Nguồn": "CoRAG multi-step",
                        "Số nguồn": len(corag_result.sources),
                        "Query dùng": question.strip(),
                        "_full_question": question.strip(),
                        "_full_answer": corag_result.answer,
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
    if st.session_state.get("enable_corag_compare") and st.session_state.get("corag_result"):
        st.markdown("#### ⚔️ So sánh RAG vs CoRAG")

        col_rag, col_corag = st.columns(2)
        with col_rag:
            st.markdown(
                '<div class="card" style="border-left:3px solid #2563EB;">'
                '<div class="card-header">🔍 RAG</div>'
                f'<div class="card-body">{st.session_state["last_answer"]}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with col_corag:
            st.markdown(
                '<div class="card" style="border-left:3px solid #10B981;">'
                '<div class="card-header">⚡ CoRAG</div>'
                f'<div class="card-body">{st.session_state["corag_result"].answer}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        raw_result = st.session_state.get("corag_result_raw")
        if isinstance(raw_result, CoRAGResult):
            m1, m2, m3 = st.columns(3)
            m1.metric("🔢 Steps", f"RAG: 1 / CoRAG: {raw_result.steps}")
            m2.metric(
                "📄 Docs",
                f"RAG: {len(st.session_state['last_sources'])} / CoRAG: {raw_result.total_docs}",
            )
            m3.metric(
                "🎯 Confidence",
                f"RAG: {st.session_state['last_confidence']:.2f} / CoRAG: {st.session_state['corag_result'].confidence:.2f}",
            )

            st.markdown("#### 🔗 CoRAG Chain Trace")
            for step in raw_result.chain:
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
    else:
        st.info(
            "💡 Bật **So sánh RAG vs CoRAG** ở sidebar và chạy câu hỏi trong tab RAG để xem so sánh."
        )


# ── TAB 3: Statistics ──
with tab_stats:
    render_statistics_tab()


# ── TAB 4: Benchmark ──
with tab_bench:
    render_benchmark_tab()

    # Handle benchmark execution with progressive display
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
