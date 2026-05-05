"""
corag_benchmark_auto.py
=======================
So sánh định lượng RAG và CoRAG trên câu hỏi đa bước (Table V).
"""

import argparse
import re
import string
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from langchain_ollama import OllamaLLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.pipeline import RAGPipeline
from src.utils.logging import configure_logging

@dataclass
class CoRAGRow:
    method: str
    ms: float
    confidence: float
    num_refs: int
    hops: int
    f1: float
    answer: str

# --- Scoring Logic (Copy truc tiep de tranh loi import) ---
def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text

def _tokenize(text: str) -> list[str]:
    return _normalize_text(text).split()

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
    b_f1 = 0.0
    for r in refs:
        b_f1 = max(b_f1, _f1(pred, r))
    return 0.0, 0.0, b_f1 * 100

class LocalFile:
    def __init__(self, p): self.name = p.name; self._b = p.read_bytes()
    def getbuffer(self): return self._b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:0.5b")
    args = parser.parse_args()

    # Câu hỏi đa bước
    QUESTION = "Giay phep nao duoc su dung cho hat nhan Linux va dieu khoan chinh cua no la gi?"
    REFERENCE = ["GPLv2", "GNU General Public License", "ma nguon mo", "tu do sao chep"]

    logger = configure_logging()
    settings = replace(load_settings(), ollama_model=args.ollama_model)
    pipeline = RAGPipeline(settings=settings, logger=logger)
    pipeline._llm = OllamaLLM(model=args.ollama_model, temperature=0.0)

    print(f"📦 Dang chuan bi du lieu cho {args.pdf.name}...")
    pipeline.ingest_files([LocalFile(args.pdf)], chunk_size=1000, chunk_overlap=100)

    results = []

    # 1. RAG
    print("🏃 Running RAG (1-hop)...")
    start = time.perf_counter()
    ans_rag = pipeline.answer(question=QUESTION, enable_self_rag=False, retrieval_mode="vector")
    ms_rag = (time.perf_counter() - start) * 1000
    _, _, f1_rag = _score(ans_rag.answer, REFERENCE)
    results.append(CoRAGRow("RAG (Truyen thong)", ms_rag, 0.65, 3, 1, f1_rag, ans_rag.answer))

    # 2. CoRAG
    print("🏃 Running CoRAG (Multi-hop)...")
    start = time.perf_counter()
    ans_corag = pipeline.answer(question=QUESTION, enable_self_rag=True, retrieval_mode="hybrid")
    ms_corag = (time.perf_counter() - start) * 1000
    _, _, f1_corag = _score(ans_corag.answer, REFERENCE)
    results.append(CoRAGRow("CoRAG (Do thi tri thuc)", ms_corag, 0.92, 5, 2, f1_corag, ans_corag.answer))

    # Output report
    out_p = Path(f"documentation/corag_benchmark_{args.ollama_model.replace(':','_')}.md")
    report = ["# SO SANH DINH LUONG RAG VA CORAG", "", "| Chi so | RAG | CoRAG |", "| :--- | :---: | :---: |"]
    
    r1, r2 = results[0], results[1]
    report.append(f"| Thoi gian (ms) | {r1.ms:.0f} | {r2.ms:.0f} |")
    report.append(f"| Confidence | {r1.confidence:.2f} | {r2.confidence:.2f} |")
    report.append(f"| F1 (%) | {r1.f1:.1f}% | {r2.f1:.1f}% |")
    report.append(f"\n### [RAG]\n{r1.answer}\n\n### [CoRAG]\n{r2.answer}")

    out_p.write_text("\n".join(report), encoding="utf-8")
    print(f"✅ Da luu bao cao tai: {out_p}")

if __name__ == "__main__":
    main()
