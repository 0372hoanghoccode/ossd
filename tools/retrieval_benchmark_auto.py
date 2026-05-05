"""
retrieval_benchmark_auto.py
==========================
Script benchmark các chế độ truy xuất (Table III):
Vector, Keyword, Hybrid, Hybrid + Rerank.
"""

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

@dataclass
class RetrievalRow:
    mode: str
    rerank: bool
    em: float
    cem: float
    f1: float
    ms: float
    answer: str

# --- Scoring Logic (Reused) ---
def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text

def _tokenize(text: str) -> list[str]:
    return _normalize_text(text).split()

def _containment_em(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not ref_tokens: return 1.0
    if not pred_tokens: return 0.0
    pred_set = {}
    for t in pred_tokens: pred_set[t] = pred_set.get(t, 0) + 1
    found = 0
    for t in ref_tokens:
        if pred_set.get(t, 0) > 0:
            found += 1
            pred_set[t] -= 1
    return found / len(ref_tokens)

def _f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens: return 1.0
    if not pred_tokens or not ref_tokens: return 0.0
    ref_counts = {}
    for t in ref_tokens: ref_counts[t] = ref_counts.get(t, 0) + 1
    common = 0
    for t in pred_tokens:
        if ref_counts.get(t, 0) > 0:
            common += 1
            ref_counts[t] -= 1
    if common == 0: return 0.0
    p = common / len(pred_tokens)
    r = common / len(ref_tokens)
    return (2 * p * r) / (p + r)

def _score(pred, refs):
    if not refs: return 0.0, 0.0, 0.0
    b_em, b_cem, b_f1 = 0.0, 0.0, 0.0
    for r in refs:
        norm_p, norm_r = _normalize_text(pred), _normalize_text(r)
        b_em = max(b_em, 1.0 if norm_p == norm_r else 0.0)
        b_cem = max(b_cem, _containment_em(pred, r))
        b_f1 = max(b_f1, _f1(pred, r))
    return b_em * 100, b_cem * 100, b_f1 * 100

class LocalFile:
    def __init__(self, p): self.name = p.name; self._b = p.read_bytes()
    def getbuffer(self): return self._b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--references", nargs="*", required=True)
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:7b")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Fixed chunk config based on Table II results
    C_SIZE, C_OVERLAP = 1000, 100
    
    modes = [
        ("vector", False, "Vector Only"),
        ("keyword", False, "Keyword (BM25)"),
        ("hybrid", False, "Hybrid (RRF)"),
        ("hybrid", True, "Hybrid + Rerank")
    ]

    logger = configure_logging()
    # Sử dụng replace vì settings là Frozen Dataclass
    settings = replace(
        load_settings(),
        ollama_model=args.ollama_model
    )
    pipeline = RAGPipeline(settings=settings, logger=logger)
    pipeline._llm = OllamaLLM(model=args.ollama_model, temperature=0.0)

    print(f"📦 Đang xử lý tài liệu: {args.pdf.name} (Size={C_SIZE}, Overlap={C_OVERLAP})...")
    pipeline.ingest_files([LocalFile(args.pdf)], chunk_size=C_SIZE, chunk_overlap=C_OVERLAP)

    results = []
    for mode_id, use_rerank, label in modes:
        print(f"🔍 Đang kiểm tra chế độ: {label}...")
        start = time.perf_counter()
        ans = pipeline.answer(
            question=args.question,
            retrieval_mode=mode_id,
            enable_rerank=use_rerank,
            conversational=False
        )
        ms = (time.perf_counter() - start) * 1000
        em, cem, f1 = _score(ans.answer, args.references)
        results.append(RetrievalRow(label, use_rerank, em, cem, f1, ms, ans.answer))

    # Build Report
    out_p = args.output or Path(f"documentation/retrieval_benchmark_{args.ollama_model.replace(':','_')}.md")
    report = [
        "# SO SÁNH CÁC CHẾ ĐỘ RETRIEVAL (Table III)",
        f"- **Model:** `{args.ollama_model}`",
        f"- **Question:** {args.question}",
        f"- **Chunk Config:** Size={C_SIZE}, Overlap={C_OVERLAP}",
        "",
        "| Chế độ | EM (%) | CEM (%) | F1 (%) | Thời gian (ms) |",
        "| :--- | ---: | ---: | ---: | ---: |"
    ]
    for r in results:
        report.append(f"| {r.mode} | {r.em:.1f} | {r.cem:.1f} | {r.f1:.1f} | {r.ms:.0f} |")
    
    report.append("\n## Mẫu câu trả lời")
    for r in results:
        report.append(f"### {r.mode}\n> {r.answer[:300]}...\n")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(report), encoding="utf-8")
    print(f"✅ Báo cáo đã lưu: {out_p}")

if __name__ == "__main__":
    main()
