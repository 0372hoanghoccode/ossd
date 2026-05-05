from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re
import string
import sys
import time
from typing import Any

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.corag_engine import CoRAGResult
from src.pipeline import RAGPipeline
from src.utils.logging import configure_logging


@dataclass
class PaperRowII:
    size: int
    overlap: int
    chunks: int
    em: float
    f1: float
    ms: float


@dataclass
class RunRowII:
    size: int
    overlap: int
    chunks: int
    em: float
    f1: float
    ms: float


@dataclass
class RunRowIII:
    mode: str
    em: float
    f1: float
    ms: float


@dataclass
class RunRowIV:
    setup: str
    em: float
    f1: float


@dataclass
class RunRowV:
    system: str
    time_ms: float
    confidence: float
    avg_docs: float
    avg_steps: float
    f1: float


@dataclass
class MetricSummary:
    em: float
    f1: float
    ms: float


class LocalUploadedFile:
    def __init__(self, file_path: Path):
        self.name = file_path.name
        self._payload = file_path.read_bytes()

    def getbuffer(self) -> bytes:
        return self._payload


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


def _score_prediction(prediction: str, references: list[str]) -> tuple[float, float]:
    if not references:
        return 0.0, 0.0

    best_em = 0.0
    best_f1 = 0.0
    for reference in references:
        best_em = max(best_em, _exact_match(prediction, reference))
        best_f1 = max(best_f1, _f1(prediction, reference))
    return best_em, best_f1


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_pdf_text(pdf_path: Path) -> str:
    with pdfplumber.open(str(pdf_path)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def _extract_table_ii(pdf_text: str) -> list[PaperRowII]:
    baseline = [
        PaperRowII(500, 50, 6412, 61.5, 68.2, 142.0),
        PaperRowII(500, 100, 6748, 63.0, 70.1, 151.0),
        PaperRowII(1000, 100, 3284, 67.5, 74.8, 128.0),
        PaperRowII(1000, 200, 3512, 66.0, 73.4, 134.0),
        PaperRowII(1500, 100, 2198, 64.5, 72.0, 118.0),
        PaperRowII(1500, 200, 2341, 65.5, 73.1, 122.0),
        PaperRowII(2000, 200, 1648, 62.0, 70.5, 109.0),
    ]

    scope_start = pdf_text.find("TableII")
    if scope_start < 0:
        scope_start = pdf_text.find("Table II")
    scope = pdf_text[scope_start : scope_start + 3400] if scope_start >= 0 else pdf_text

    pattern = re.compile(
        r"(500|1000|1500|2000)\s+(50|100|200)\s+(\d{3,6})\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+(\d{2,4})"
    )
    rows: list[PaperRowII] = []
    seen: set[tuple[int, int]] = set()
    for match in pattern.finditer(scope):
        size = int(match.group(1))
        overlap = int(match.group(2))
        key = (size, overlap)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            PaperRowII(
                size=size,
                overlap=overlap,
                chunks=int(match.group(3)),
                em=float(match.group(4)),
                f1=float(match.group(5)),
                ms=float(match.group(6)),
            )
        )

    if len(rows) >= 5:
        wanted_order = [(item.size, item.overlap) for item in baseline]
        row_by_key = {(item.size, item.overlap): item for item in rows}
        ordered: list[PaperRowII] = []
        for key in wanted_order:
            if key in row_by_key:
                ordered.append(row_by_key[key])
        return ordered if ordered else baseline

    return baseline


def _extract_table_iii(pdf_text: str) -> dict[str, tuple[float, float, float]]:
    baseline = {
        "Vector only": (59.5, 67.3, 95.0),
        "Keyword (BM25)": (55.0, 62.8, 48.0),
        "Hybrid (RRF)": (67.5, 74.8, 143.0),
        "Hybrid + Rerank": (71.0, 78.3, 312.0),
    }

    compact = re.sub(r"\s+", " ", pdf_text)
    pattern = re.compile(
        r"Vectoronly\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)"
        r".*?Keyword\(BM25\)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)"
        r".*?Hybrid\(RRF\)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)"
        r".*?Hybrid\+Rerank\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(compact)
    if not match:
        return baseline

    return {
        "Vector only": (float(match.group(1)), float(match.group(2)), float(match.group(3))),
        "Keyword (BM25)": (float(match.group(4)), float(match.group(5)), float(match.group(6))),
        "Hybrid (RRF)": (float(match.group(7)), float(match.group(8)), float(match.group(9))),
        "Hybrid + Rerank": (float(match.group(10)), float(match.group(11)), float(match.group(12))),
    }


def _extract_table_iv(pdf_text: str) -> dict[str, tuple[float, float]]:
    baseline = {
        "RAG single-turn (no history)": (48.3, 55.7),
        "RAG multi-turn (no rewrite)": (58.5, 65.2),
        "RAG multi-turn + query rewrite": (71.7, 77.4),
    }

    compact = re.sub(r"\s+", " ", pdf_text)
    pattern = re.compile(
        r"RAGđơnlượt\(khônghistory\)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)"
        r".*?RAGđalượt\(khôngrewrite\)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)"
        r".*?RAGđalượt\+queryrewrite\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(compact)
    if not match:
        return baseline

    return {
        "RAG single-turn (no history)": (float(match.group(1)), float(match.group(2))),
        "RAG multi-turn (no rewrite)": (float(match.group(3)), float(match.group(4))),
        "RAG multi-turn + query rewrite": (float(match.group(5)), float(match.group(6))),
    }


def _load_eval_set(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _make_pipeline(base_settings: Any, logger: Any, ollama_model: str, retriever_k: int) -> RAGPipeline:
    settings = replace(base_settings, ollama_model=ollama_model, retriever_k=max(1, retriever_k))
    return RAGPipeline(settings=settings, logger=logger)


def _ingest_docs(pipeline: RAGPipeline, pdf_paths: list[Path], chunk_size: int, chunk_overlap: int) -> int:
    uploads = [LocalUploadedFile(path) for path in pdf_paths]
    result = pipeline.ingest_files(uploads, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return result.chunks


def _run_rag_eval(
    pipeline: RAGPipeline,
    questions: list[dict[str, Any]],
    retrieval_mode: str,
    enable_rerank: bool,
    conversational: bool,
    enable_query_rewrite: bool,
    use_history: bool,
) -> MetricSummary:
    em_scores: list[float] = []
    f1_scores: list[float] = []
    time_ms_values: list[float] = []

    for item in questions:
        question = str(item.get("question", "")).strip()
        references = [str(text) for text in item.get("references", [])]
        history = item.get("history", []) if use_history else []

        started = time.perf_counter()
        answer = pipeline.answer(
            question=question,
            chat_history=history,
            retrieval_mode=retrieval_mode,
            metadata_filters=None,
            enable_rerank=enable_rerank,
            enable_self_rag=False,
            conversational=conversational,
            enable_query_rewrite=enable_query_rewrite,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        em, f1 = _score_prediction(answer.answer, references)
        em_scores.append(em * 100.0)
        f1_scores.append(f1 * 100.0)
        time_ms_values.append(elapsed_ms)

    return MetricSummary(em=_mean(em_scores), f1=_mean(f1_scores), ms=_mean(time_ms_values))


def _run_table_v_eval(
    pipeline: RAGPipeline,
    questions: list[dict[str, Any]],
    retrieval_mode: str,
    enable_rerank: bool,
    corag_max_steps: int,
) -> tuple[RunRowV, RunRowV]:
    rag_times: list[float] = []
    rag_conf: list[float] = []
    rag_docs: list[float] = []
    rag_f1: list[float] = []

    corag_times: list[float] = []
    corag_conf: list[float] = []
    corag_docs: list[float] = []
    corag_steps: list[float] = []
    corag_f1: list[float] = []

    for item in questions:
        question = str(item.get("question", "")).strip()
        references = [str(text) for text in item.get("references", [])]

        rag_started = time.perf_counter()
        rag = pipeline.answer(
            question=question,
            chat_history=None,
            retrieval_mode=retrieval_mode,
            metadata_filters=None,
            enable_rerank=enable_rerank,
            enable_self_rag=False,
            conversational=False,
            enable_query_rewrite=False,
        )
        rag_elapsed = (time.perf_counter() - rag_started) * 1000.0
        _, rag_f1_score = _score_prediction(rag.answer, references)

        rag_times.append(rag_elapsed)
        rag_conf.append(rag.confidence)
        rag_docs.append(float(len(rag.sources)))
        rag_f1.append(rag_f1_score * 100.0)

        corag_started = time.perf_counter()
        corag = pipeline.answer_corag(
            question=question,
            max_steps=max(2, corag_max_steps),
            retrieval_mode=retrieval_mode,
            metadata_filters=None,
            enable_rerank=enable_rerank,
            step_callback=None,
        )
        corag_elapsed = (time.perf_counter() - corag_started) * 1000.0
        _, corag_f1_score = _score_prediction(corag.answer, references)

        raw = pipeline._last_corag_result
        if isinstance(raw, CoRAGResult):
            corag_docs.append(float(raw.total_docs))
            corag_steps.append(float(raw.steps))
        else:
            corag_docs.append(float(len(corag.sources)))
            corag_steps.append(1.0)

        corag_times.append(corag_elapsed)
        corag_conf.append(corag.confidence)
        corag_f1.append(corag_f1_score * 100.0)

    rag_row = RunRowV(
        system="RAG",
        time_ms=_mean(rag_times),
        confidence=_mean(rag_conf),
        avg_docs=_mean(rag_docs),
        avg_steps=1.0,
        f1=_mean(rag_f1),
    )
    corag_row = RunRowV(
        system="CoRAG",
        time_ms=_mean(corag_times),
        confidence=_mean(corag_conf),
        avg_docs=_mean(corag_docs),
        avg_steps=_mean(corag_steps),
        f1=_mean(corag_f1),
    )
    return rag_row, corag_row


def _build_report(
    generated_at: str,
    pdf_path: Path,
    docs: list[Path],
    paper_ii: list[PaperRowII],
    run_ii: list[RunRowII],
    paper_iii: dict[str, tuple[float, float, float]],
    run_iii: list[RunRowIII],
    paper_iv: dict[str, tuple[float, float]],
    run_iv: list[RunRowIV],
    run_v: list[RunRowV],
) -> str:
    lines: list[str] = []
    lines.append("# Full PDF Metrics Reproduction")
    lines.append("")
    lines.append(f"Generated at: {generated_at}")
    lines.append(f"PDF source: {pdf_path}")
    lines.append("Docs used:")
    for item in docs:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Table II - Chunk Strategy")
    lines.append("| Size | Overlap | Paper Chunks | Run Chunks | Paper EM | Run EM | Paper F1 | Run F1 | Paper ms | Run ms |")
    lines.append("| ---: | ------: | -----------: | ---------: | -------: | -----: | -------: | -----: | -------: | -----: |")

    run_ii_lookup = {(row.size, row.overlap): row for row in run_ii}
    for paper in paper_ii:
        run = run_ii_lookup[(paper.size, paper.overlap)]
        lines.append(
            f"| {paper.size} | {paper.overlap} | {paper.chunks} | {run.chunks} | {paper.em:.1f} | {run.em:.1f} | {paper.f1:.1f} | {run.f1:.1f} | {paper.ms:.1f} | {run.ms:.1f} |"
        )

    lines.append("")
    lines.append("## Table III - Retrieval Modes")
    lines.append("| Mode | Paper EM | Run EM | Paper F1 | Run F1 | Paper ms | Run ms |")
    lines.append("| :--- | -------: | -----: | -------: | -----: | -------: | -----: |")
    for run in run_iii:
        paper = paper_iii.get(run.mode)
        if paper is None:
            lines.append(f"| {run.mode} | - | {run.em:.1f} | - | {run.f1:.1f} | - | {run.ms:.1f} |")
            continue
        lines.append(
            f"| {run.mode} | {paper[0]:.1f} | {run.em:.1f} | {paper[1]:.1f} | {run.f1:.1f} | {paper[2]:.1f} | {run.ms:.1f} |"
        )

    lines.append("")
    lines.append("## Table IV - Query Rewrite in Multi-turn RAG")
    lines.append("| Setup | Paper EM | Run EM | Paper F1 | Run F1 |")
    lines.append("| :---- | -------: | -----: | -------: | -----: |")
    for run in run_iv:
        paper = paper_iv.get(run.setup)
        if paper is None:
            lines.append(f"| {run.setup} | - | {run.em:.1f} | - | {run.f1:.1f} |")
            continue
        lines.append(f"| {run.setup} | {paper[0]:.1f} | {run.em:.1f} | {paper[1]:.1f} | {run.f1:.1f} |")

    lines.append("")
    lines.append("## Table V - Quantitative RAG vs CoRAG")
    lines.append("| System | Avg time (ms) | Avg confidence | Avg docs | Avg steps | F1 |")
    lines.append("| :----- | -------------: | -------------: | -------: | --------: | --: |")
    for row in run_v:
        lines.append(
            f"| {row.system} | {row.time_ms:.1f} | {row.confidence:.2f} | {row.avg_docs:.2f} | {row.avg_steps:.2f} | {row.f1:.1f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- EM and F1 are computed from local answer-level QA evaluation using data/real_docs/pdf_eval_set.json.")
    lines.append("- The paper used a different test set, model, and hardware, so absolute values are expected to differ.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiments that map to numeric tables in the project PDF report.")
    parser.add_argument("--paper-pdf", type=Path, default=Path(".pdf"))
    parser.add_argument("--docs-dir", type=Path, default=Path("data/real_docs/downloads"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/real_docs/pdf_eval_set.json"))
    parser.add_argument("--output", type=Path, default=Path("documentation/chunk_strategy_comparison_report.md"))
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:0.5b")
    parser.add_argument("--retriever-k", type=int, default=3)
    parser.add_argument("--table2-retrieval-mode", type=str, default="hybrid", choices=["vector", "keyword", "hybrid"])
    parser.add_argument("--table5-retrieval-mode", type=str, default="hybrid", choices=["vector", "keyword", "hybrid"])
    parser.add_argument("--table5-max-steps", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pdf_path = args.paper_pdf.resolve()
    docs_dir = args.docs_dir.resolve()
    eval_set_path = args.eval_set.resolve()
    output_path = args.output.resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF file: {pdf_path}")
    if not docs_dir.exists():
        raise FileNotFoundError(f"Missing docs directory: {docs_dir}")
    if not eval_set_path.exists():
        raise FileNotFoundError(f"Missing eval set file: {eval_set_path}")

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No pdf files found in: {docs_dir}")

    logger = configure_logging()
    base_settings = load_settings()
    eval_set = _load_eval_set(eval_set_path)

    pdf_text = _read_pdf_text(pdf_path)
    paper_ii = _extract_table_ii(pdf_text)
    paper_iii = _extract_table_iii(pdf_text)
    paper_iv = _extract_table_iv(pdf_text)

    run_ii: list[RunRowII] = []
    for paper in paper_ii:
        pipeline = _make_pipeline(base_settings, logger, args.ollama_model, args.retriever_k)
        chunks = _ingest_docs(pipeline, pdf_files, paper.size, paper.overlap)
        summary = _run_rag_eval(
            pipeline=pipeline,
            questions=list(eval_set.get("table2_questions", [])),
            retrieval_mode=args.table2_retrieval_mode,
            enable_rerank=False,
            conversational=False,
            enable_query_rewrite=False,
            use_history=False,
        )
        run_ii.append(RunRowII(size=paper.size, overlap=paper.overlap, chunks=chunks, em=summary.em, f1=summary.f1, ms=summary.ms))

    # Use paper-best chunk config for tables III/IV/V.
    best_paper = max(paper_ii, key=lambda item: item.f1)
    pipeline_main = _make_pipeline(base_settings, logger, args.ollama_model, args.retriever_k)
    _ingest_docs(pipeline_main, pdf_files, best_paper.size, best_paper.overlap)

    run_iii: list[RunRowIII] = []
    for mode_name, mode, rerank in [
        ("Vector only", "vector", False),
        ("Keyword (BM25)", "keyword", False),
        ("Hybrid (RRF)", "hybrid", False),
        ("Hybrid + Rerank", "hybrid", True),
    ]:
        summary = _run_rag_eval(
            pipeline=pipeline_main,
            questions=list(eval_set.get("table3_questions", [])),
            retrieval_mode=mode,
            enable_rerank=rerank,
            conversational=False,
            enable_query_rewrite=False,
            use_history=False,
        )
        run_iii.append(RunRowIII(mode=mode_name, em=summary.em, f1=summary.f1, ms=summary.ms))

    run_iv: list[RunRowIV] = []
    table4_questions = list(eval_set.get("table4_dialogues", []))

    summary_single = _run_rag_eval(
        pipeline=pipeline_main,
        questions=table4_questions,
        retrieval_mode="hybrid",
        enable_rerank=False,
        conversational=False,
        enable_query_rewrite=False,
        use_history=False,
    )
    run_iv.append(RunRowIV(setup="RAG single-turn (no history)", em=summary_single.em, f1=summary_single.f1))

    summary_multi_no_rewrite = _run_rag_eval(
        pipeline=pipeline_main,
        questions=table4_questions,
        retrieval_mode="hybrid",
        enable_rerank=False,
        conversational=True,
        enable_query_rewrite=False,
        use_history=True,
    )
    run_iv.append(
        RunRowIV(setup="RAG multi-turn (no rewrite)", em=summary_multi_no_rewrite.em, f1=summary_multi_no_rewrite.f1)
    )

    summary_multi_rewrite = _run_rag_eval(
        pipeline=pipeline_main,
        questions=table4_questions,
        retrieval_mode="hybrid",
        enable_rerank=False,
        conversational=True,
        enable_query_rewrite=True,
        use_history=True,
    )
    run_iv.append(
        RunRowIV(setup="RAG multi-turn + query rewrite", em=summary_multi_rewrite.em, f1=summary_multi_rewrite.f1)
    )

    rag_row_v, corag_row_v = _run_table_v_eval(
        pipeline=pipeline_main,
        questions=list(eval_set.get("table5_questions", [])),
        retrieval_mode=args.table5_retrieval_mode,
        enable_rerank=False,
        corag_max_steps=args.table5_max_steps,
    )

    report = _build_report(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        pdf_path=pdf_path,
        docs=pdf_files,
        paper_ii=paper_ii,
        run_ii=run_ii,
        paper_iii=paper_iii,
        run_iii=run_iii,
        paper_iv=paper_iv,
        run_iv=run_iv,
        run_v=[rag_row_v, corag_row_v],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Report written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
