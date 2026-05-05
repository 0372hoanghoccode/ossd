from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import Settings


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove extra spaces and punctuation."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def calculate_em(predicted: str, reference: str) -> float:
    """Calculate Exact Match score."""
    return 1.0 if normalize_text(predicted) == normalize_text(reference) else 0.0


def calculate_cem(predicted: str, reference: str) -> float:
    """Calculate Character-level Exact Match (CEM) - percentage of matching characters."""
    pred_norm = normalize_text(predicted)
    ref_norm = normalize_text(reference)
    
    if not ref_norm:
        return 1.0 if not pred_norm else 0.0
    
    matches = sum(1 for a, b in zip(pred_norm, ref_norm) if a == b)
    max_len = max(len(pred_norm), len(ref_norm))
    
    return matches / max_len if max_len > 0 else 0.0


def calculate_f1(predicted: str, reference: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = set(normalize_text(predicted).split())
    ref_tokens = set(normalize_text(reference).split())
    
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    if not pred_tokens:
        return 0.0
    
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

RUN_HISTORY_PATH = Path("data/run_history.json")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #2563EB;
    --primary-light: #DBEAFE;
    --primary-lighter: #EFF6FF;
    --primary-dark: #1D4ED8;
    --accent: #3B82F6;
    --text-main: #1E293B;
    --text-secondary: #64748B;
    --bg-white: #FFFFFF;
    --bg-page: #F0F7FF;
    --border: #BFDBFE;
    --border-light: #E0EFFE;
    --radius: 12px;
    --shadow-sm: 0 1px 3px rgba(37,99,235,.08);
    --shadow-md: 0 4px 14px rgba(37,99,235,.12);
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-page) !important;
    color: var(--text-main) !important;
}
[data-testid="stApp"] { background: var(--bg-page) !important; }
#MainMenu, footer { visibility: hidden; }
header { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, var(--primary-lighter) 100%) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0.5rem !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--primary) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.2rem !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    margin-bottom: 0.2rem !important;
}
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stMultiSelect {
    margin-bottom: 0.15rem !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
    background: var(--bg-white) !important;
    color: var(--primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    font-size: 0.82rem !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover {
    background: var(--primary-light) !important;
    border-color: var(--primary) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    background: var(--bg-white) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 2px solid var(--border) !important;
    background: var(--bg-white) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 0 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.65rem 1.4rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.2s ease !important;
    font-size: 0.9rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
    font-weight: 700 !important;
    background: var(--primary-lighter) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-white) !important;
    border-radius: 0 0 12px 12px !important;
    border: 1px solid var(--border-light) !important;
    border-top: none !important;
    padding: 1rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    padding: 0.8rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-weight: 700 !important;
}

/* ── Cards ── */
.card {
    background: var(--bg-white);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow-sm);
}
.card-header {
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
}
.card-body {
    color: var(--text-main);
    font-size: 0.85rem;
    line-height: 1.55;
}

mark {
    background: #BFDBFE;
    border-radius: 3px;
    padding: 0.1em 0.2em;
}

.badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-ready { background: #DCFCE7; color: #166534; }
.badge-pending { background: #FEF3C7; color: #92400E; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.8rem !important;
    background: var(--bg-white) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
}

/* ── Selects ── */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    background: var(--bg-white) !important;
    min-height: 32px !important;
}

[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

hr { border-color: var(--border-light) !important; }

.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: var(--text-main) !important;
    background: var(--primary-lighter) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 0.8rem !important;
}
</style>
"""


def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar(settings: Settings, pipeline_ready: bool, indexed_chunks: int = 0) -> None:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;margin-bottom:0.4rem;'>"
            "<span style='font-size:1.6rem;'>📄</span><br>"
            "<span style='font-size:1.1rem;font-weight:700;color:#2563EB;'>SmartDoc AI</span><br>"
            "<span style='color:#64748B;font-size:.75rem;'>Hỏi đáp tài liệu thông minh</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        if pipeline_ready:
            st.markdown(
                '<span class="badge badge-ready">✅ Sẵn sàng</span>'
                f'&nbsp;<span style="color:#64748B;font-size:.75rem;">{indexed_chunks} chunks</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge badge-pending">⏳ Chưa có dữ liệu</span>',
                unsafe_allow_html=True,
            )

        with st.expander("ℹ️ Hệ thống", expanded=False):
            st.caption(f"**LLM:** {settings.ollama_model}")
            st.caption(f"**Embed:** {settings.embedding_model}")
            st.caption(f"**k={settings.retriever_k}** | chunk={settings.chunk_size}/{settings.chunk_overlap}")


def _collect_highlight_terms(*texts: str, max_terms: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for term in re.findall(r"\w+", text or "", flags=re.UNICODE):
            normalized = term.strip().lower()
            if (len(normalized) < 3 and not normalized.isdigit()) or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(normalized)
            if len(terms) >= max_terms:
                return terms
    return terms


def _highlight_query_terms(text: str, query: str, answer: str = "") -> str:
    escaped_text = html.escape(text)
    terms = _collect_highlight_terms(query)
    if not terms:
        return escaped_text
    pattern = re.compile("(" + "|".join(re.escape(term) for term in terms) + ")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped_text)


def render_source_preview(query: str = "", answer: str = "") -> None:
    if not st.session_state.get("last_sources"):
        return

    st.markdown("#### 📎 Nguồn tham chiếu")
    sources = st.session_state["last_sources"]

    doc_groups: dict[str, list] = {}
    for doc in sources:
        source_name = doc.metadata.get("source_name", doc.metadata.get("source", "?"))
        doc_groups.setdefault(source_name, []).append(doc)

    colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
    badge_html_parts: list[str] = []
    for i, (name, group) in enumerate(doc_groups.items()):
        color = colors[i % len(colors)]
        badge_html_parts.append(
            f'<span style="display:inline-block;background:{color}15;color:{color};'
            f'border:1px solid {color}40;padding:0.25rem 0.7rem;border-radius:20px;'
            f'font-size:0.78rem;font-weight:600;margin-right:0.4rem;margin-bottom:0.4rem;">'
            f'📄 {html.escape(name)} ({len(group)} chunks)</span>'
        )
    st.markdown(
        f'<div style="margin-bottom:0.5rem">{"".join(badge_html_parts)}</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(sources), 4))
    for idx, doc in enumerate(sources[:4]):
        source_name = doc.metadata.get("source_name", doc.metadata.get("source", "?"))
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        start_index = doc.metadata.get("start_index", "?")
        color = colors[list(doc_groups.keys()).index(source_name) % len(colors)] if source_name in doc_groups else "#2563EB"
        with cols[idx]:
            st.markdown(
                f'<div class="card" style="border-left:3px solid {color};">'
                f'<div class="card-header" style="color:{color}">#{idx + 1} {html.escape(str(source_name))}</div>'
                f'<div class="card-body">Trang {page} · chunk {chunk_id} · offset {start_index}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    with st.expander("📖 Xem nội dung nguồn tham chiếu", expanded=True):
        options = []
        for idx, doc in enumerate(sources, start=1):
            source_name = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            start_index = doc.metadata.get("start_index", "?")
            options.append(f"#{idx} | {source_name} | trang {page} | chunk {chunk_id} | offset {start_index}")

        selected = st.selectbox("Chọn đoạn để xem", options=options, label_visibility="collapsed")
        selected_doc = sources[options.index(selected)]
        source_name = selected_doc.metadata.get("source_name", selected_doc.metadata.get("source", "unknown"))
        page = selected_doc.metadata.get("page", "?")
        chunk_id = selected_doc.metadata.get("chunk_id", "?")
        start_index = selected_doc.metadata.get("start_index", "?")
        retrieval_score = selected_doc.metadata.get("retrieval_score")
        rerank_score = selected_doc.metadata.get("rerank_score")
        doc_type = selected_doc.metadata.get("doc_type", "?")
        uploaded_at = selected_doc.metadata.get("uploaded_at", "?")
        highlighted = _highlight_query_terms(selected_doc.page_content, query, answer)

        meta_bits = [f"Trang {page}", f"Chunk {chunk_id}", f"Offset {start_index}"]
        if retrieval_score is not None:
            meta_bits.append(f"Retrieval {float(retrieval_score):.3f}")
        if rerank_score is not None:
            meta_bits.append(f"Rerank {float(rerank_score):.3f}")
        st.caption(f"{source_name} | " + " | ".join(meta_bits))

        st.markdown(
            f'<div style="margin-bottom:0.5rem;">'
            f'<span class="badge badge-ready">Loại: {html.escape(str(doc_type).upper())}</span> '
            f'<span class="badge" style="background:#DBEAFE;color:#2563EB;">Upload: {html.escape(str(uploaded_at))}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="card"><div class="card-body">{highlighted}</div></div>',
            unsafe_allow_html=True,
        )


def render_document_manager(indexed_docs: list[dict]) -> None:
    """Render the document management panel showing all indexed documents with metadata."""
    if not indexed_docs:
        st.info("📭 Chưa có tài liệu nào được index. Upload và xử lý PDF/DOCX để bắt đầu.")
        return

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">'
        f'<span style="font-size:1.3rem;">📚</span>'
        f'<span style="font-size:1rem;font-weight:600;color:#1E293B;">'
        f'{len(indexed_docs)} tài liệu đã index</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
    for i, doc_info in enumerate(indexed_docs):
        color = colors[i % len(colors)]
        file_name = doc_info.get("Tên file", "?")
        doc_type = doc_info.get("Loại", "?")
        uploaded = doc_info.get("Ngày upload", "?")
        chunks = doc_info.get("Số chunks", 0)
        pages = doc_info.get("Số trang", 0)

        type_icon = "📕" if doc_type.upper() == "PDF" else "📘" if doc_type.upper() == "DOCX" else "📄"

        st.markdown(
            f'<div class="card" style="border-left:4px solid {color};padding:0.8rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<div style="font-weight:600;color:{color};font-size:0.9rem;">{type_icon} {html.escape(str(file_name))}</div>'
            f'<div style="color:#64748B;font-size:0.75rem;margin-top:0.2rem;">Upload: {html.escape(str(uploaded))}</div>'
            f'</div>'
            f'<div style="display:flex;gap:0.6rem;">'
            f'<div style="text-align:center;">'
            f'<div style="font-weight:700;color:{color};font-size:1rem;">{chunks}</div>'
            f'<div style="font-size:0.65rem;color:#94A3B8;">chunks</div>'
            f'</div>'
            f'<div style="text-align:center;">'
            f'<div style="font-weight:700;color:{color};font-size:1rem;">{pages}</div>'
            f'<div style="font-size:0.65rem;color:#94A3B8;">trang</div>'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📊 Xem dạng bảng", expanded=False):
        st.dataframe(indexed_docs, use_container_width=True, hide_index=True)


def render_chat_history(history: list[dict[str, str]]) -> None:
    with st.sidebar:
        if not history:
            return

        st.markdown("---")
        st.markdown("#### 💬 Lịch sử")

        recent_history = list(reversed(history[-10:]))
        history_options = [
            f"#{len(history) - offset} | {item['question'][:50]}{'...' if len(item['question']) > 50 else ''}"
            for offset, item in enumerate(recent_history)
        ]
        selected_option = st.selectbox(
            "Xem lại câu hỏi đã hỏi",
            options=history_options,
            key="history_review_select",
        )
        selected_item = recent_history[history_options.index(selected_option)]

        st.markdown(
            f'<div class="card" style="padding:.6rem;">'
            f'<div style="font-weight:600;font-size:.78rem;color:#2563EB;">Q: {html.escape(selected_item["question"][:80])}</div>'
            f'<div style="font-size:.72rem;color:#64748B;margin-top:.2rem;">{html.escape(selected_item["answer"][:100])}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Dùng lại câu hỏi", key="reuse_history_question", use_container_width=True):
            st.session_state["qa_question"] = selected_item["question"]


# ── Statistics & Benchmark helpers ──

BENCHMARK_HISTORY_PATH = Path("data/benchmark_history.json")
BENCHMARK_RUNS_DIR = Path("data/benchmark_runs")


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text to be used as filename."""
    # Remove special characters, keep only alphanumeric, spaces, and Vietnamese characters
    text = re.sub(r'[^\w\s\-áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', text.lower())
    # Replace spaces with underscores
    text = re.sub(r'\s+', '_', text.strip())
    # Limit length
    return text[:max_length]


def save_run_to_history(entry: dict) -> None:
    """Append a single run entry to data/run_history.json immediately."""
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if RUN_HISTORY_PATH.exists():
        try:
            history = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            history = []
    entry["#"] = len(history) + 1
    entry["Thời điểm"] = datetime.now().strftime("%H:%M:%S")
    history.append(entry)
    RUN_HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_run_history() -> list[dict]:
    """Load all run history entries."""
    if not RUN_HISTORY_PATH.exists():
        return []
    try:
        return json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_benchmark_entry(entry: dict, benchmark_session_id: str = None) -> None:
    """Save a single benchmark result to a session-specific file."""
    if benchmark_session_id:
        # Save to session-specific file
        session_file = BENCHMARK_RUNS_DIR / f"{benchmark_session_id}.json"
        BENCHMARK_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        
        history: list[dict] = []
        if session_file.exists():
            try:
                history = json.loads(session_file.read_text(encoding="utf-8"))
            except Exception:
                history = []
        
        entry["#"] = len(history) + 1
        entry["Thời điểm"] = datetime.now().strftime("%H:%M:%S")
        entry["Ngày"] = datetime.now().strftime("%Y-%m-%d")
        history.append(entry)
        session_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # Fallback to old behavior
        BENCHMARK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        if BENCHMARK_HISTORY_PATH.exists():
            try:
                history = json.loads(BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8"))
            except Exception:
                history = []
        entry["#"] = len(history) + 1
        entry["Thời điểm"] = datetime.now().strftime("%H:%M:%S")
        entry["Ngày"] = datetime.now().strftime("%Y-%m-%d")
        history.append(entry)
        BENCHMARK_HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_benchmark_history() -> list[dict]:
    """Load all benchmark entries from all session files."""
    all_history = []
    
    # Load from benchmark_runs directory
    if BENCHMARK_RUNS_DIR.exists():
        for session_file in sorted(BENCHMARK_RUNS_DIR.glob("*.json"), reverse=True):
            try:
                session_data = json.loads(session_file.read_text(encoding="utf-8"))
                # Add session info to each entry
                for entry in session_data:
                    entry["Session"] = session_file.stem
                all_history.extend(session_data)
            except Exception:
                continue
    
    # Also load from old benchmark_history.json if exists
    if BENCHMARK_HISTORY_PATH.exists():
        try:
            old_history = json.loads(BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8"))
            for entry in old_history:
                entry["Session"] = "legacy"
            all_history.extend(old_history)
        except Exception:
            pass
    
    return all_history


def list_benchmark_sessions() -> list[dict]:
    """List all benchmark session files with metadata."""
    sessions = []
    
    if BENCHMARK_RUNS_DIR.exists():
        for session_file in sorted(BENCHMARK_RUNS_DIR.glob("*.json"), reverse=True):
            try:
                session_data = json.loads(session_file.read_text(encoding="utf-8"))
                if session_data:
                    first_entry = session_data[0]
                    sessions.append({
                        "file": session_file.name,
                        "session_id": session_file.stem,
                        "question": first_entry.get("question", ""),
                        "date": first_entry.get("Ngày", ""),
                        "time": first_entry.get("Thời điểm", ""),
                        "count": len(session_data),
                    })
            except Exception:
                continue
    
    return sessions


def render_statistics_tab() -> None:
    """Render the statistics tab with filters and run history."""

    history = load_run_history()
    if not history:
        st.info("📭 Chưa có dữ liệu")
        return

    # Reference answer input at the top
   
    eval_ref_answer = st.text_area(
        "Đáp án tham chiếu",
        key="stats_eval_ref_answer",
        placeholder="Nhập đáp án để tính EM, CEM, F1...",
        height=70
    )

    # ── Filters ──
    st.markdown("**🔎 Bộ lọc**")

    # Get all unique values for filters
    all_questions = sorted({str(h.get("Câu hỏi", "?")) for h in history})
    all_modes = sorted({h.get("Mode", "?") for h in history})
    all_conv = sorted({h.get("Conversational", "?") for h in history})
    all_rerank = sorted({h.get("Rerank", "?") for h in history})
    all_selfrag = sorted({h.get("Self-RAG", "?") for h in history})

    # All filters in one row
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)

    with fc1:
        sel_questions = st.multiselect("Câu hỏi", options=all_questions, default=[], key="stat_filter_question")
    with fc2:
        sel_modes = st.multiselect("Mode", options=all_modes, default=[], key="stat_filter_mode")
    with fc3:
        sel_conv = st.multiselect("Conv", options=all_conv, default=[], key="stat_filter_conv")
    with fc4:
        sel_rerank = st.multiselect("Rerank", options=all_rerank, default=[], key="stat_filter_rerank")
    with fc5:
        sel_selfrag = st.multiselect("Self-RAG", options=all_selfrag, default=[], key="stat_filter_selfrag")

    # Apply filters
    filtered = history
    if sel_questions:
        filtered = [h for h in filtered if str(h.get("Câu hỏi", "?")) in sel_questions]
    if sel_modes:
        filtered = [h for h in filtered if h.get("Mode") in sel_modes]
    if sel_conv:
        filtered = [h for h in filtered if h.get("Conversational") in sel_conv]
    if sel_rerank:
        filtered = [h for h in filtered if h.get("Rerank") in sel_rerank]
    if sel_selfrag:
        filtered = [h for h in filtered if h.get("Self-RAG") in sel_selfrag]

    # Calculate metrics if reference answer provided
    has_ref_answer = eval_ref_answer.strip() != ""
    if has_ref_answer:
        for h in filtered:
            answer = h.get("_full_answer", "")
            if answer:
                h["_em"] = calculate_em(answer, eval_ref_answer)
                h["_cem"] = calculate_cem(answer, eval_ref_answer)
                h["_f1"] = calculate_f1(answer, eval_ref_answer)
            else:
                h["_em"] = 0.0
                h["_cem"] = 0.0
                h["_f1"] = 0.0

    # Summary metrics
    total_runs = len(filtered)
    avg_gen = 0
    modes_count: dict[str, int] = {}
    avg_em = 0.0
    avg_cem = 0.0
    avg_f1 = 0.0
    
    for h in filtered:
        try:
            gen_val = str(h.get("Generation (ms)", "0")).replace(",", "").replace("—", "0")
            avg_gen += int(gen_val)
        except (ValueError, TypeError):
            pass
        mode = h.get("Mode", "?")
        modes_count[mode] = modes_count.get(mode, 0) + 1
        
        if has_ref_answer:
            avg_em += h.get("_em", 0.0)
            avg_cem += h.get("_cem", 0.0)
            avg_f1 += h.get("_f1", 0.0)

    avg_gen = avg_gen // max(total_runs, 1)
    top_mode = max(modes_count, key=modes_count.get) if modes_count else "?"
    
    if has_ref_answer and total_runs > 0:
        avg_em = avg_em / total_runs
        avg_cem = avg_cem / total_runs
        avg_f1 = avg_f1 / total_runs

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔢 Kết quả", f"{total_runs}/{len(history)}")
    c2.metric("⚡ Avg Gen", f"{avg_gen:,} ms")
    c3.metric("🏆 Top Mode", top_mode)
    if has_ref_answer:
        c4.metric("🎯 Avg F1", f"{avg_f1:.2f}")
    else:
        c4.metric("📄 Tổng gốc", len(history))

    # Show filtered table with action buttons
    st.markdown("**📋 Kết quả**")
    
    # Add custom CSS for compact rows
    st.markdown("""
    <style>
    div[data-testid="column"] > div {
        font-size: 0.85rem;
        padding: 0.2rem 0;
    }
    .stButton > button {
        padding: 0.2rem 0.4rem;
        font-size: 0.75rem;
        min-height: 1.8rem;
        height: 1.8rem;
        line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if filtered:
        # Create table header
        cols = st.columns([0.4, 1.8, 0.7, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5])
        cols[0].markdown("**#**")
        cols[1].markdown("**Câu hỏi**")
        cols[2].markdown("**Mode**")
        cols[3].markdown("**Conv**")
        cols[4].markdown("**Rerank**")
        cols[5].markdown("**Self**")
        cols[6].markdown("**Conf**")
        cols[7].markdown("**Retr**")
        cols[8].markdown("**Gen**")
        cols[9].markdown("**EM**")
        cols[10].markdown("**CEM**")
        cols[11].markdown("**F1**")
        cols[12].markdown("**🔍**")
        cols[13].markdown("**🗑️**")
        
        st.divider()
        
        # Data rows
        for i, h in enumerate(filtered):
            row_cols = st.columns([0.4, 1.8, 0.7, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5])
            
            row_cols[0].markdown(f"<small>{h.get('#', '?')}</small>", unsafe_allow_html=True)
            row_cols[1].markdown(f"<small>{str(h.get('Câu hỏi', '?'))[:35]}...</small>", unsafe_allow_html=True)
            row_cols[2].markdown(f"<small>{h.get('Mode', '?')}</small>", unsafe_allow_html=True)
            row_cols[3].markdown(f"<small>{h.get('Conversational', '?')}</small>", unsafe_allow_html=True)
            row_cols[4].markdown(f"<small>{h.get('Rerank', '?')}</small>", unsafe_allow_html=True)
            row_cols[5].markdown(f"<small>{h.get('Self-RAG', '?')}</small>", unsafe_allow_html=True)
            row_cols[6].markdown(f"<small>{h.get('Confidence', '?')}</small>", unsafe_allow_html=True)
            row_cols[7].markdown(f"<small>{str(h.get('Retrieval (ms)', '?'))}</small>", unsafe_allow_html=True)
            row_cols[8].markdown(f"<small>{str(h.get('Generation (ms)', '?'))}</small>", unsafe_allow_html=True)
            
            # Metrics columns
            if has_ref_answer:
                row_cols[9].markdown(f"<small>{h.get('_em', 0.0):.2f}</small>", unsafe_allow_html=True)
                row_cols[10].markdown(f"<small>{h.get('_cem', 0.0):.2f}</small>", unsafe_allow_html=True)
                row_cols[11].markdown(f"<small>{h.get('_f1', 0.0):.2f}</small>", unsafe_allow_html=True)
            else:
                row_cols[9].markdown("<small>0.00</small>", unsafe_allow_html=True)
                row_cols[10].markdown("<small>0.00</small>", unsafe_allow_html=True)
                row_cols[11].markdown("<small>0.00</small>", unsafe_allow_html=True)
            
            # Detail button
            if row_cols[12].button("🔍", key=f"detail_stat_{i}", help="Chi tiết"):
                st.session_state[f"show_detail_{i}"] = not st.session_state.get(f"show_detail_{i}", False)
            
            # Delete button
            if row_cols[13].button("🗑️", key=f"del_stat_{i}", help="Xóa"):
                st.session_state["open_delete_stat_dialog"] = True
                st.session_state["delete_stat_index"] = h.get("#", -1)
                st.rerun()
            # Show detail if toggled
            if st.session_state.get(f"show_detail_{i}", False):
                st.markdown(
                    f'<div style="background:#EFF6FF;border-left:3px solid #2563EB;padding:0.8rem;margin:0.5rem 0;border-radius:4px;">'
                    f'<b>📝 Câu hỏi đầy đủ:</b><br>{html.escape(str(h.get("_full_question", h.get("Câu hỏi", "?"))))}<br><br>'
                    f'<b>💬 Câu trả lời:</b><br>{html.escape(str(h.get("_full_answer", "N/A"))[:500])}{"..." if len(str(h.get("_full_answer", ""))) > 500 else ""}<br><br>'
                    f'<b>🎯 Confidence:</b> {h.get("Confidence", "?")} | '
                    f'<b>📄 Số nguồn:</b> {h.get("Số nguồn", "?")} | '
                    f'<b>🔎 Query:</b> {html.escape(str(h.get("Query dùng", "?")))}<br>'
                    f'<b>📚 Nguồn:</b> {html.escape(str(h.get("Nguồn", "?")))}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            if i < len(filtered) - 1:
                st.divider()
    else:
        st.info("📭 Không có kết quả phù hợp với bộ lọc")


def render_benchmark_tab() -> None:
    """Render the benchmark tab — run chunk experiments with progressive display."""

    bc1, bc2 = st.columns(2)
    with bc1:
        bm_sizes = st.multiselect(
            "chunk_size",
            options=[500, 1000, 1500, 2000],
            default=[500, 1000, 1500, 2000],
            key="bm_chunk_sizes",
        )
    with bc2:
        bm_overlaps = st.multiselect(
            "chunk_overlap",
            options=[50, 100, 200],
            default=[50, 100, 200],
            key="bm_chunk_overlaps",
        )

    bm_question = st.text_input("📝 Câu hỏi benchmark", key="bm_question", placeholder="Nhập câu hỏi để test...")
    bm_reference_answer = st.text_area("✅ Đáp án tham chiếu (Reference Answer)", key="bm_reference_answer", 
                                        placeholder="Nhập đáp án đúng để tính EM, CEM, F1...", height=100)

    total_configs = len(bm_sizes) * len(bm_overlaps)
    st.caption(f"📦 Tổng: **{total_configs}** cấu hình sẽ chạy")

    # Run button — actual execution happens in app.py
    if st.button("🚀 Chạy Benchmark", key="run_benchmark_tab", type="primary",
                  disabled=total_configs == 0 or not bm_question.strip()):
        # Create session ID based on question and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_slug = sanitize_filename(bm_question.strip(), max_length=30)
        session_id = f"{timestamp}_{question_slug}"
        
        st.session_state["bm_should_run"] = True
        st.session_state["bm_configs"] = [(s, o) for s in bm_sizes for o in bm_overlaps]
        st.session_state["bm_q"] = bm_question.strip()
        st.session_state["bm_ref_answer"] = bm_reference_answer.strip()
        st.session_state["bm_session_id"] = session_id

    # Show benchmark sessions
    st.divider()
    st.markdown("##### 📋 Các lần chạy Benchmark")
    
    sessions = list_benchmark_sessions()
    if sessions:
        st.caption(f"Tổng: **{len(sessions)}** lần chạy")
        
        # Display sessions as cards
        for session in sessions[:10]:  # Show latest 10
            with st.expander(f"📁 {session['date']} {session['time']} - {session['question'][:50]}...", expanded=False):
                st.caption(f"**Session ID:** {session['session_id']}")
                st.caption(f"**File:** {session['file']}")
                st.caption(f"**Số configs:** {session['count']}")
                
                # Load and display this session's data
                session_file = BENCHMARK_RUNS_DIR / session['file']
                try:
                    session_data = json.loads(session_file.read_text(encoding="utf-8"))
                    display_bm = []
                    for h in session_data:
                        row = {k: v for k, v in h.items() if k not in ["answer", "reference_answer", "Session"]}
                        display_bm.append(row)
                    st.dataframe(display_bm, use_container_width=True, hide_index=True)
                    
                    # Show answers
                    for h in session_data[:3]:  # Show first 3
                        ans = h.get("answer", "")[:200]
                        ref = h.get("reference_answer", "")
                        st.markdown(
                            f'<div style="background:#fff;border-left:3px solid #F59E0B;padding:0.5rem;margin:0.3rem 0;border-radius:4px;">'
                            f'<b>#{h.get("#", "?")} | size={h.get("chunk_size")} overlap={h.get("chunk_overlap")}</b><br>'
                            f'<small><b>A:</b> {html.escape(ans)}{"..." if len(ans) >= 200 else ""}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")
        
        if st.button("🗑️ Xóa tất cả benchmark", key="clear_all_bm"):
            st.session_state["open_clear_bm_dialog"] = True
    else:
        st.info("📭 Chưa có lịch sử benchmark. Chạy benchmark ở trên để bắt đầu.")
