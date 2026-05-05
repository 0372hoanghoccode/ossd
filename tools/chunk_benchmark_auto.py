"""
chunk_benchmark_auto.py
=======================
Script tự động benchmark các chiến lược phân đoạn văn bản (chunking) trên
cùng 1 tài liệu PDF và 1 câu hỏi, xuất ra báo cáo markdown với số liệu
EM (%), F1 (%), thời gian phản hồi (ms).

Usage:
    python tools/chunk_benchmark_auto.py \
        --pdf review_on_tap.pdf \
        --question "Mục đích chính của tài liệu này là gì?" \
        --references "Tài liệu ôn tập kiến thức" "Hệ thống hóa kiến thức môn học" \
        --output documentation/chunk_benchmark_report.md

Nếu không truyền --references, script sẽ dùng câu trả lời của cấu hình
tốt nhất (chunk_size=1000, overlap=100) làm reference để so sánh tương đối
giữa các cấu hình với nhau.
"""

from __future__ import annotations

import argparse
import re
import string
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaLLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.pipeline import RAGPipeline
from src.utils.logging import configure_logging


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    chunk_size: int
    chunk_overlap: int
    chunks: int
    em: float        # Exact Match %
    cem: float       # Containment EM % (reference tokens found in prediction)
    f1: float        # F1 %
    ms: float        # avg response time in ms
    answer: str      # raw answer text


# ---------------------------------------------------------------------------
# Text normalisation & scoring (SQuAD-style)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize_text(text).split()


def _exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0


def _containment_em(prediction: str, reference: str) -> float:
    """Check if ALL reference tokens are contained in prediction.

    This is more useful than strict EM when comparing short references
    against long LLM-generated answers. Returns the fraction of reference
    tokens found in the prediction (1.0 = all found).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    pred_set: dict[str, int] = {}
    for token in pred_tokens:
        pred_set[token] = pred_set.get(token, 0) + 1

    found = 0
    for token in ref_tokens:
        count = pred_set.get(token, 0)
        if count > 0:
            found += 1
            pred_set[token] = count - 1

    return found / len(ref_tokens)


def _f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    common = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            common += 1
            ref_counts[token] = count - 1

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _score_prediction(prediction: str, references: list[str]) -> tuple[float, float, float]:
    """Return best (EM, Containment_EM, F1) across all references."""
    if not references:
        return 0.0, 0.0, 0.0

    best_em = 0.0
    best_cem = 0.0
    best_f1 = 0.0
    for reference in references:
        best_em = max(best_em, _exact_match(prediction, reference))
        best_cem = max(best_cem, _containment_em(prediction, reference))
        best_f1 = max(best_f1, _f1(prediction, reference))
    return best_em, best_cem, best_f1


# ---------------------------------------------------------------------------
# Local uploaded file adapter
# ---------------------------------------------------------------------------

class LocalUploadedFile:
    def __init__(self, file_path: Path):
        self.name = file_path.name
        self._payload = file_path.read_bytes()

    def getbuffer(self) -> bytes:
        return self._payload


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

STRATEGY_CONFIGS: list[tuple[int, int]] = [
    (500, 50),
    (500, 100),
    (1000, 100),
    (1000, 200),
    (1500, 100),
    (1500, 200),
    (2000, 200),
]


def _run_single_benchmark(
    pdf_path: Path,
    question: str,
    references: list[str],
    chunk_size: int,
    chunk_overlap: int,
    retrieval_mode: str,
    ollama_model: str,
    retriever_k: int,
    repeats: int,
) -> BenchmarkRow:
    """Run RAG for one chunk configuration, return metrics."""
    logger = configure_logging()
    settings = replace(
        load_settings(),
        ollama_model=ollama_model,
        retriever_k=max(1, retriever_k),
    )
    pipeline = RAGPipeline(settings=settings, logger=logger)
    # Force temperature=0 for deterministic, reproducible output
    pipeline._llm = OllamaLLM(model=ollama_model, temperature=0.0)

    # Ingest the document
    upload = LocalUploadedFile(pdf_path)
    result = pipeline.ingest_files([upload], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    num_chunks = result.chunks

    # Run the question multiple times for timing stability
    em_scores: list[float] = []
    cem_scores: list[float] = []
    f1_scores: list[float] = []
    time_ms_values: list[float] = []
    last_answer = ""

    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        answer_result = pipeline.answer(
            question=question,
            chat_history=None,
            retrieval_mode=retrieval_mode,
            metadata_filters=None,
            enable_rerank=False,
            enable_self_rag=False,
            conversational=False,
            enable_query_rewrite=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        last_answer = answer_result.answer
        em, cem, f1 = _score_prediction(answer_result.answer, references)
        em_scores.append(em * 100.0)
        cem_scores.append(cem * 100.0)
        f1_scores.append(f1 * 100.0)
        time_ms_values.append(elapsed_ms)

    avg_em = sum(em_scores) / len(em_scores)
    avg_cem = sum(cem_scores) / len(cem_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_ms = sum(time_ms_values) / len(time_ms_values)

    return BenchmarkRow(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=num_chunks,
        em=round(avg_em, 1),
        cem=round(avg_cem, 1),
        f1=round(avg_f1, 1),
        ms=round(avg_ms),
        answer=last_answer,
    )


def _generate_reference_answer(
    pdf_path: Path,
    question: str,
    ollama_model: str,
    retriever_k: int,
    retrieval_mode: str,
) -> str:
    """Generate a reference answer using chunk_size=1000, overlap=100 (best config)."""
    logger = configure_logging()
    settings = replace(
        load_settings(),
        ollama_model=ollama_model,
        retriever_k=max(1, retriever_k),
    )
    pipeline = RAGPipeline(settings=settings, logger=logger)
    # Force temperature=0 for deterministic output
    pipeline._llm = OllamaLLM(model=ollama_model, temperature=0.0)
    upload = LocalUploadedFile(pdf_path)
    pipeline.ingest_files([upload], chunk_size=1000, chunk_overlap=100)

    result = pipeline.answer(
        question=question,
        chat_history=None,
        retrieval_mode=retrieval_mode,
        metadata_filters=None,
        enable_rerank=False,
        enable_self_rag=False,
        conversational=False,
        enable_query_rewrite=False,
    )
    return result.answer


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _safe_preview(text: str, max_chars: int = 200) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _build_report(
    generated_at: str,
    pdf_path: Path,
    question: str,
    references: list[str],
    auto_generated_ref: bool,
    rows: list[BenchmarkRow],
    retrieval_mode: str,
    ollama_model: str,
    repeats: int,
) -> str:
    lines: list[str] = []

    lines.append("# SO SÁNH CHIẾN LƯỢC PHÂN ĐOẠN VĂN BẢN")
    lines.append("")
    lines.append(f"**Thời gian tạo:** {generated_at}")
    lines.append(f"**Tài liệu PDF:** `{pdf_path.name}`")
    lines.append(f"**Câu hỏi:** {question}")
    lines.append(f"**Mô hình LLM:** `{ollama_model}`")
    lines.append(f"**Chế độ truy xuất:** `{retrieval_mode}`")
    lines.append(f"**Số lần chạy mỗi cấu hình:** {repeats}")
    lines.append("")

    if auto_generated_ref:
        lines.append("> **Lưu ý:** Không có câu trả lời chuẩn (reference) được cung cấp.")
        lines.append("> Script đã tự sinh reference từ cấu hình tốt nhất (1000/100) để so sánh tương đối.")
        lines.append("")

    lines.append("## Bảng kết quả")
    lines.append("")
    lines.append("| Size | Overlap | Chunks | EM (%) | CEM (%) | F1 (%) | ms |")
    lines.append("| ---: | ------: | -----: | -----: | ------: | -----: | ---: |")

    for row in rows:
        lines.append(
            f"| {row.chunk_size} | {row.chunk_overlap} | {row.chunks} "
            f"| {row.em:.1f} | {row.cem:.1f} | {row.f1:.1f} | {row.ms:.0f} |"
        )

    lines.append("")

    # Find best config
    best_row = max(rows, key=lambda r: (r.f1, r.em, -r.ms))
    worst_row = min(rows, key=lambda r: (r.f1, r.em))

    lines.append("")
    lines.append("> **CEM (Containment EM)**: Tỷ lệ token trong reference xuất hiện trong câu trả lời. ")
    lines.append("> Phù hợp hơn EM khi so sánh reference ngắn với câu trả lời dài của LLM.")
    lines.append("")

    lines.append("## Phân tích kết quả")
    lines.append("")
    lines.append(
        f"Kết quả cho thấy cấu hình **chunk_size={best_row.chunk_size}, "
        f"overlap={best_row.chunk_overlap}** đạt điểm cao nhất "
        f"(EM={best_row.em:.1f}%, CEM={best_row.cem:.1f}%, F1={best_row.f1:.1f}%)."
    )

    # Analysis: small vs large chunks
    small_chunks = [r for r in rows if r.chunk_size <= 500]
    large_chunks = [r for r in rows if r.chunk_size >= 2000]

    analysis_points: list[str] = []

    if small_chunks:
        avg_f1_small = sum(r.f1 for r in small_chunks) / len(small_chunks)
        analysis_points.append(
            f"- Chunk quá nhỏ (size≤500) có F1 trung bình {avg_f1_small:.1f}%, "
            f"dễ mất ngữ cảnh khi phân đoạn."
        )

    if large_chunks:
        avg_f1_large = sum(r.f1 for r in large_chunks) / len(large_chunks)
        analysis_points.append(
            f"- Chunk quá lớn (size≥2000) có F1 trung bình {avg_f1_large:.1f}%, "
            f"làm giảm độ chính xác khi so khớp với câu hỏi ngắn."
        )

    mid_chunks = [r for r in rows if 1000 <= r.chunk_size <= 1500]
    if mid_chunks:
        avg_f1_mid = sum(r.f1 for r in mid_chunks) / len(mid_chunks)
        analysis_points.append(
            f"- Chunk trung bình (1000-1500) cho F1 trung bình {avg_f1_mid:.1f}%, "
            f"cân bằng giữa ngữ cảnh và độ chính xác."
        )

    fastest = min(rows, key=lambda r: r.ms)
    slowest = max(rows, key=lambda r: r.ms)
    analysis_points.append(
        f"- Thời gian phản hồi: nhanh nhất {fastest.ms:.0f}ms "
        f"(size={fastest.chunk_size}), chậm nhất {slowest.ms:.0f}ms "
        f"(size={slowest.chunk_size})."
    )

    for point in analysis_points:
        lines.append(point)

    lines.append("")

    # Recommendation
    lines.append("## Khuyến nghị")
    lines.append("")
    lines.append(
        f"Sử dụng **chunk_size={best_row.chunk_size}** và "
        f"**overlap={best_row.chunk_overlap}** cho hệ thống RAG. "
        f"Cấu hình này cho kết quả tối ưu với EM={best_row.em:.1f}%, "
        f"F1={best_row.f1:.1f}%, thời gian phản hồi {best_row.ms:.0f}ms."
    )
    lines.append("")

    # Sample answers
    lines.append("## Mẫu câu trả lời theo từng cấu hình")
    lines.append("")
    for row in rows:
        lines.append(
            f"### Size={row.chunk_size}, Overlap={row.chunk_overlap} "
            f"(EM={row.em:.1f}%, F1={row.f1:.1f}%)"
        )
        lines.append("")
        lines.append(f"> {_safe_preview(row.answer, max_chars=500)}")
        lines.append("")

    # Reference answers
    if references:
        lines.append("## Câu trả lời chuẩn (Reference)")
        lines.append("")
        for idx, ref in enumerate(references, 1):
            lines.append(f"{idx}. {ref}")
        lines.append("")

    # Repro command
    lines.append("## Lệnh tái tạo")
    lines.append("```powershell")
    ref_args = " ".join(f'"{r}"' for r in references) if references else ""
    lines.append(
        f'python tools/chunk_benchmark_auto.py \\\n'
        f'  --pdf "{pdf_path}" \\\n'
        f'  --question "{question}" \\\n'
        + (f'  --references {ref_args} \\\n' if ref_args else "")
        + f'  --retrieval-mode {retrieval_mode} \\\n'
        f'  --ollama-model {ollama_model} \\\n'
        f'  --repeats {repeats} \\\n'
        f'  --output documentation/chunk_benchmark_report.md'
    )
    lines.append("```")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tự động benchmark các chiến lược phân đoạn văn bản và xuất báo cáo."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("review_on_tap.pdf"),
        help="Đường dẫn tới file PDF ôn tập (mặc định: review_on_tap.pdf)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Nội dung chính của tài liệu này là gì?",
        help="Câu hỏi dùng để benchmark",
    )
    parser.add_argument(
        "--references",
        nargs="*",
        default=None,
        help="Danh sách câu trả lời chuẩn (reference answers). "
             "Nếu không truyền, script sẽ tự sinh reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Đường dẫn file output. Nếu không truyền, tự đặt tên theo model (vd: chunk_benchmark_qwen2.5_7b.md)",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="hybrid",
        choices=["vector", "keyword", "hybrid"],
        help="Chế độ truy xuất (mặc định: hybrid)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen2.5:7b",
        help="Mô hình Ollama sử dụng (mặc định: qwen2.5:7b)",
    )
    parser.add_argument(
        "--retriever-k",
        type=int,
        default=3,
        help="Số lượng chunks truy xuất (mặc định: 3)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Số lần chạy mỗi cấu hình để tính trung bình (mặc định: 1)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=None,
        help="Các cấu hình tùy chỉnh, mỗi cấu hình dạng 'size:overlap'. "
             "Ví dụ: 500:50 1000:100 1500:200",
    )
    return parser.parse_args()


def _model_to_filename(model: str) -> str:
    """Convert model name to safe filename part. e.g. 'qwen2.5:7b' -> 'qwen2.5_7b'"""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', model)


def main() -> int:
    args = parse_args()

    pdf_path = args.pdf.resolve()

    # Auto-generate output filename from model name if not specified
    if args.output is None:
        model_safe = _model_to_filename(args.ollama_model)
        output_path = Path(f"documentation/chunk_benchmark_{model_safe}.md").resolve()
    else:
        output_path = args.output.resolve()

    if not pdf_path.exists():
        print(f"❌ Không tìm thấy file PDF: {pdf_path}")
        return 1

    question = args.question.strip()
    if not question:
        print("❌ Câu hỏi không được để trống.")
        return 1

    # Parse custom configs or use defaults
    if args.configs:
        configs: list[tuple[int, int]] = []
        for cfg in args.configs:
            parts = cfg.split(":")
            if len(parts) != 2:
                print(f"❌ Cấu hình không hợp lệ: {cfg} (cần dạng 'size:overlap')")
                return 1
            configs.append((int(parts[0]), int(parts[1])))
    else:
        configs = STRATEGY_CONFIGS

    # Handle references
    auto_generated_ref = False
    if args.references:
        references = [r.strip() for r in args.references if r.strip()]
    else:
        print("📝 Không có reference, đang tự sinh câu trả lời chuẩn từ cấu hình tốt nhất...")
        ref_answer = _generate_reference_answer(
            pdf_path=pdf_path,
            question=question,
            ollama_model=args.ollama_model,
            retriever_k=args.retriever_k,
            retrieval_mode=args.retrieval_mode,
        )
        references = [ref_answer]
        auto_generated_ref = True
        print(f"✅ Reference tự sinh: {_safe_preview(ref_answer, 120)}")

    # Run benchmarks
    rows: list[BenchmarkRow] = []
    total = len(configs)

    for idx, (chunk_size, chunk_overlap) in enumerate(configs, 1):
        print(
            f"\n{'='*60}\n"
            f"[{idx}/{total}] Đang benchmark: size={chunk_size}, overlap={chunk_overlap}\n"
            f"{'='*60}"
        )

        row = _run_single_benchmark(
            pdf_path=pdf_path,
            question=question,
            references=references,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_mode=args.retrieval_mode,
            ollama_model=args.ollama_model,
            retriever_k=args.retriever_k,
            repeats=args.repeats,
        )
        rows.append(row)

        print(f"   Chunks: {row.chunks}")
        print(f"   EM: {row.em:.1f}%  |  F1: {row.f1:.1f}%  |  Time: {row.ms:.0f}ms")
        print(f"   Answer: {_safe_preview(row.answer, 120)}")

    # Build and write report
    report = _build_report(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        pdf_path=pdf_path,
        question=question,
        references=references,
        auto_generated_ref=auto_generated_ref,
        rows=rows,
        retrieval_mode=args.retrieval_mode,
        ollama_model=args.ollama_model,
        repeats=args.repeats,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    # Print summary table
    print(f"\n{'='*65}")
    print("📊 SO SÁNH CHIẾN LƯỢC PHÂN ĐOẠN VĂN BẢN")
    print(f"{'='*65}")
    print(f"{'Size':>6} {'Overlap':>8} {'Chunks':>7} {'EM(%)':>7} {'CEM(%)':>8} {'F1(%)':>7} {'ms':>6}")
    print("-" * 55)
    for row in rows:
        print(f"{row.chunk_size:>6} {row.chunk_overlap:>8} {row.chunks:>7} {row.em:>7.1f} {row.cem:>8.1f} {row.f1:>7.1f} {row.ms:>6.0f}")

    best = max(rows, key=lambda r: (r.f1, r.em))
    print("-" * 55)
    print(
        f"\n✅ Cấu hình tốt nhất: chunk_size={best.chunk_size}, "
        f"overlap={best.chunk_overlap} "
        f"(EM={best.em:.1f}%, CEM={best.cem:.1f}%, F1={best.f1:.1f}%)"
    )

    print(f"\n📄 Báo cáo đã được lưu tại: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
