from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import statistics
import re
import sys
import textwrap
from typing import Any

import pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.corag_engine import CoRAGStep
from src.pipeline import RAGPipeline
from src.utils.logging import configure_logging


@dataclass
class PaperRow:
    chunk_size: int
    chunk_overlap: int
    chunks: int
    em: float
    f1: float
    ms: int


@dataclass
class RunRow:
    chunk_size: int
    chunk_overlap: int
    chunks: int
    split_ms_avg: float
    split_ms_std: float
    em_proxy: float
    f1_proxy: float
    benchmark_queries: int


@dataclass
class CoragTraceReport:
    question: str
    rag_time_s: float
    corag_time_s: float
    rag_confidence: float
    corag_confidence: float
    corag_steps: int
    corag_docs: int
    steps: list[CoRAGStep]
    rag_answer_preview: str
    corag_answer_preview: str


class LocalUploadedFile:
    def __init__(self, file_path: Path):
        self.name = file_path.name
        self._payload = file_path.read_bytes()

    def getbuffer(self) -> bytes:
        return self._payload


def _extract_table_ii(pdf_path: Path) -> list[PaperRow]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        full_text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    index = full_text.find("TableII")
    if index == -1:
        index = full_text.find("Table II")

    search_scope = full_text[index : index + 3200] if index >= 0 else full_text
    pattern = re.compile(
        r"(500|1000|1500|2000)\s+"
        r"(50|100|200)\s+"
        r"(\d{3,6})\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+"
        r"(\d{2,4})"
    )

    seen: set[tuple[int, int]] = set()
    rows: list[PaperRow] = []

    for match in pattern.finditer(search_scope):
        chunk_size = int(match.group(1))
        chunk_overlap = int(match.group(2))
        key = (chunk_size, chunk_overlap)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            PaperRow(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunks=int(match.group(3)),
                em=float(match.group(4)),
                f1=float(match.group(5)),
                ms=int(match.group(6)),
            )
        )

    preferred_order = [
        (500, 50),
        (500, 100),
        (1000, 100),
        (1000, 200),
        (1500, 100),
        (1500, 200),
        (2000, 200),
    ]

    if rows:
        by_key = {(row.chunk_size, row.chunk_overlap): row for row in rows}
        ordered: list[PaperRow] = []
        for key in preferred_order:
            if key in by_key:
                ordered.append(by_key[key])
        for row in rows:
            key = (row.chunk_size, row.chunk_overlap)
            if key not in preferred_order:
                ordered.append(row)
        return ordered

    raise ValueError(f"Could not parse Table II rows from PDF: {pdf_path}")


def _load_pdf_documents(pdf_paths: list[Path]) -> list[Document]:
    all_docs: list[Document] = []
    for path in pdf_paths:
        loader = PDFPlumberLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("source_name", path.name)
            doc.metadata.setdefault("doc_type", "pdf")
        all_docs.extend(docs)
    return all_docs


def _run_single_config(
    pipeline: RAGPipeline,
    raw_docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    repeats: int,
    retrieval_k: int,
) -> RunRow:
    effective_repeats = max(1, repeats)
    split_ms_runs: list[float] = []
    split_docs: list[Document] = []

    for _ in range(effective_repeats):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        start = datetime.now().timestamp()
        current_split_docs = splitter.split_documents(raw_docs)
        end = datetime.now().timestamp()

        split_ms_runs.append((end - start) * 1000.0)
        split_docs = current_split_docs

    for idx, doc in enumerate(split_docs, start=1):
        doc.metadata.setdefault("chunk_id", idx)
        doc.metadata.setdefault("start_index", -1)

    em_proxy, benchmark_queries = pipeline._estimate_chunk_recall_at_k(split_docs, top_k=1)
    f1_proxy, benchmark_queries_k = pipeline._estimate_chunk_recall_at_k(split_docs, top_k=max(1, retrieval_k))

    queries = max(benchmark_queries, benchmark_queries_k)

    return RunRow(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=len(split_docs),
        split_ms_avg=statistics.mean(split_ms_runs),
        split_ms_std=statistics.pstdev(split_ms_runs) if len(split_ms_runs) > 1 else 0.0,
        em_proxy=em_proxy * 100.0,
        f1_proxy=f1_proxy * 100.0,
        benchmark_queries=queries,
    )


def _safe_preview(text: str, max_chars: int = 260) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _run_corag_trace(
    settings_model: str,
    retrieval_mode: str,
    enable_rerank: bool,
    docs: list[Path],
    question: str,
    chunk_size: int,
    chunk_overlap: int,
    max_steps: int,
) -> CoragTraceReport | None:
    logger = configure_logging()
    settings = replace(load_settings(), ollama_model=settings_model)
    pipeline = RAGPipeline(settings=settings, logger=logger)

    uploads = [LocalUploadedFile(path) for path in docs]
    pipeline.ingest_files(uploads, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    rag_result = pipeline.answer(
        question=question,
        retrieval_mode=retrieval_mode,
        enable_rerank=enable_rerank,
        conversational=False,
        enable_self_rag=False,
        metadata_filters=None,
        chat_history=None,
    )

    live_steps: list[CoRAGStep] = []

    def _capture_step(step: CoRAGStep) -> None:
        live_steps.append(step)

    corag_result = pipeline.answer_corag(
        question=question,
        max_steps=max_steps,
        retrieval_mode=retrieval_mode,
        metadata_filters=None,
        enable_rerank=enable_rerank,
        step_callback=_capture_step,
    )

    raw = pipeline._last_corag_result
    corag_docs = raw.total_docs if raw else len(corag_result.sources)
    corag_steps = raw.steps if raw else len(live_steps)

    return CoragTraceReport(
        question=question,
        rag_time_s=rag_result.retrieval_seconds + rag_result.generation_seconds,
        corag_time_s=corag_result.retrieval_seconds + corag_result.generation_seconds,
        rag_confidence=rag_result.confidence,
        corag_confidence=corag_result.confidence,
        corag_steps=corag_steps,
        corag_docs=corag_docs,
        steps=live_steps,
        rag_answer_preview=_safe_preview(rag_result.answer),
        corag_answer_preview=_safe_preview(corag_result.answer),
    )


def _build_report_markdown(
    generated_at: str,
    paper_pdf: Path,
    docs: list[Path],
    retrieval_k: int,
    repeats: int,
    paper_rows: list[PaperRow],
    run_rows: list[RunRow],
    corag_trace: CoragTraceReport | None,
    corag_error: str | None,
) -> str:
    run_lookup = {(row.chunk_size, row.chunk_overlap): row for row in run_rows}

    lines: list[str] = []
    lines.append("# Chunk Strategy Comparison Report")
    lines.append("")
    lines.append(f"Generated at: {generated_at}")
    lines.append(f"Paper PDF source: {paper_pdf}")
    lines.append("PDF corpus used for local run:")
    for doc in docs:
        lines.append(f"- {doc}")
    lines.append("")
    lines.append("## Method Notes")
    lines.append(f"- Local split timing uses {repeats} run(s) and reports average milliseconds.")
    lines.append(
        "- EM_proxy is retrieval Recall@1 (%), and F1_proxy is retrieval Recall@k (%) from pseudo-queries. "
        "These are retrieval-grounding proxies, not final answer-level EM/F1 from a labeled QA set."
    )
    lines.append(f"- k for F1_proxy is set to retriever_k = {retrieval_k}.")
    lines.append("")
    lines.append("## Paper Table II vs Local Run")
    lines.append("")
    lines.append(
        "| Size | Overlap | Paper Chunks | Paper EM (%) | Paper F1 (%) | Paper ms | Run Chunks | Run EM_proxy (%) | Run F1_proxy (%) | Run ms avg | Delta Chunks | Delta ms |"
    )
    lines.append(
        "| ---: | ------: | -----------: | -----------: | -----------: | -------: | ---------: | ---------------: | ---------------: | ---------: | -----------: | -------: |"
    )

    for paper in paper_rows:
        run = run_lookup.get((paper.chunk_size, paper.chunk_overlap))
        if run is None:
            lines.append(
                f"| {paper.chunk_size} | {paper.chunk_overlap} | {paper.chunks} | {paper.em:.1f} | {paper.f1:.1f} | {paper.ms} | - | - | - | - | - | - |"
            )
            continue

        delta_chunks = run.chunks - paper.chunks
        delta_ms = run.split_ms_avg - paper.ms

        lines.append(
            "| "
            f"{paper.chunk_size} | {paper.chunk_overlap} | {paper.chunks} | {paper.em:.1f} | {paper.f1:.1f} | {paper.ms} | "
            f"{run.chunks} | {run.em_proxy:.2f} | {run.f1_proxy:.2f} | {run.split_ms_avg:.2f} | {delta_chunks:+d} | {delta_ms:+.2f} |"
        )

    lines.append("")

    best_paper = max(paper_rows, key=lambda row: row.f1)
    best_run = max(run_rows, key=lambda row: row.f1_proxy)

    lines.append("## Key Findings")
    lines.append(
        f"- Paper best config by F1: size={best_paper.chunk_size}, overlap={best_paper.chunk_overlap}, "
        f"F1={best_paper.f1:.1f}%, EM={best_paper.em:.1f}%."
    )
    lines.append(
        f"- Local best config by F1_proxy: size={best_run.chunk_size}, overlap={best_run.chunk_overlap}, "
        f"F1_proxy={best_run.f1_proxy:.2f}%, EM_proxy={best_run.em_proxy:.2f}%."
    )

    run_for_paper_best = run_lookup.get((best_paper.chunk_size, best_paper.chunk_overlap))
    if run_for_paper_best:
        lines.append(
            f"- On paper-best config ({best_paper.chunk_size}/{best_paper.chunk_overlap}), "
            f"local split time={run_for_paper_best.split_ms_avg:.2f} ms and chunks={run_for_paper_best.chunks}."
        )

    lines.append("")
    lines.append("## CoRAG Step Trace (Local Demonstration)")
    if corag_trace is not None:
        lines.append(f"Question: {corag_trace.question}")
        lines.append(
            f"- RAG time={corag_trace.rag_time_s:.2f}s, CoRAG time={corag_trace.corag_time_s:.2f}s, "
            f"RAG confidence={corag_trace.rag_confidence:.2f}, CoRAG confidence={corag_trace.corag_confidence:.2f}."
        )
        lines.append(f"- CoRAG steps={corag_trace.corag_steps}, docs accumulated={corag_trace.corag_docs}.")
        lines.append(f"- RAG answer preview: {corag_trace.rag_answer_preview}")
        lines.append(f"- CoRAG answer preview: {corag_trace.corag_answer_preview}")
        lines.append("")
        lines.append("| Step | Query | Retrieved | Sufficient | Missing Parts | Next Query | Decision Note |")
        lines.append("| ---: | :---- | --------: | :--------: | :------------ | :--------- | :------------ |")
        for step in corag_trace.steps:
            missing = ", ".join(step.missing_parts) if step.missing_parts else "-"
            next_query = step.next_query or "-"
            note = _safe_preview(step.reasoning, max_chars=120)
            query = _safe_preview(step.query, max_chars=120)
            lines.append(
                f"| {step.step} | {query} | {step.retrieved_count} | {'yes' if step.sufficient else 'no'} | "
                f"{missing} | {next_query} | {note} |"
            )
    else:
        lines.append("- CoRAG trace was not generated.")
        if corag_error:
            lines.append(f"- Reason: {corag_error}")

    lines.append("")
    lines.append("## Repro Commands")
    lines.append("```powershell")
    lines.append(
        textwrap.dedent(
            """
            d:/Desktop/ossd/.venv/Scripts/python.exe tools/chunk_strategy_report.py \
              --paper-pdf ./.pdf \
              --docs-dir ./data/real_docs/downloads \
              --output ./documentation/chunk_strategy_comparison_report.md \
              --repeats 3 \
              --retriever-k 3
            """
        ).strip()
    )
    lines.append("```")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paper chunk strategy table with local benchmark run.")
    parser.add_argument("--paper-pdf", type=Path, default=Path(".pdf"), help="Path to report PDF containing Table II.")
    parser.add_argument("--docs-dir", type=Path, default=Path("data/real_docs/downloads"), help="Directory containing PDF corpus.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("documentation/chunk_strategy_comparison_report.md"),
        help="Output markdown report path.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="How many runs for split timing per config.")
    parser.add_argument("--retriever-k", type=int, default=3, help="k used for F1_proxy (Recall@k).")
    parser.add_argument(
        "--run-corag-trace",
        action="store_true",
        help="Also run one RAG vs CoRAG demo and include step trace in report.",
    )
    parser.add_argument(
        "--trace-question",
        type=str,
        default="How does patch management guidance in 800-40 support risk reduction goals in 800-53?",
        help="Question for CoRAG trace demo.",
    )
    parser.add_argument("--trace-max-steps", type=int, default=4, help="Max CoRAG steps for trace demo.")
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen2.5:0.5b",
        help="Ollama model used for CoRAG trace demo.",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="hybrid",
        choices=["vector", "keyword", "hybrid"],
        help="Retrieval mode for CoRAG trace demo.",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable cross-encoder reranking for CoRAG trace demo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    paper_pdf = args.paper_pdf.resolve()
    docs_dir = args.docs_dir.resolve()
    output_path = args.output.resolve()

    if not paper_pdf.exists():
        raise FileNotFoundError(f"Paper PDF not found: {paper_pdf}")
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in: {docs_dir}")

    logger = configure_logging()
    settings = replace(load_settings(), retriever_k=max(1, args.retriever_k))
    pipeline = RAGPipeline(settings=settings, logger=logger)

    paper_rows = _extract_table_ii(paper_pdf)
    raw_docs = _load_pdf_documents(pdf_paths)

    run_rows: list[RunRow] = []
    for row in paper_rows:
        run_rows.append(
            _run_single_config(
                pipeline=pipeline,
                raw_docs=raw_docs,
                chunk_size=row.chunk_size,
                chunk_overlap=row.chunk_overlap,
                repeats=args.repeats,
                retrieval_k=settings.retriever_k,
            )
        )

    corag_trace: CoragTraceReport | None = None
    corag_error: str | None = None
    if args.run_corag_trace:
        try:
            best_run = max(run_rows, key=lambda item: item.f1_proxy)
            corag_trace = _run_corag_trace(
                settings_model=args.ollama_model,
                retrieval_mode=args.retrieval_mode,
                enable_rerank=args.enable_rerank,
                docs=pdf_paths,
                question=args.trace_question,
                chunk_size=best_run.chunk_size,
                chunk_overlap=best_run.chunk_overlap,
                max_steps=max(2, args.trace_max_steps),
            )
        except Exception as exc:
            corag_error = str(exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report_markdown(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        paper_pdf=paper_pdf,
        docs=pdf_paths,
        retrieval_k=settings.retriever_k,
        repeats=max(1, args.repeats),
        paper_rows=paper_rows,
        run_rows=run_rows,
        corag_trace=corag_trace,
        corag_error=corag_error,
    )
    output_path.write_text(report, encoding="utf-8")

    print(f"Report written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
