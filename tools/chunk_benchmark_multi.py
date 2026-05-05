from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.pipeline import RAGPipeline
from src.utils.logging import configure_logging


@dataclass
class QuestionSpec:
    question_id: str
    question: str
    references: list[str]
    hypothesis: str
    rationale: str


@dataclass
class QuestionRun:
    question_id: str
    chunk_size: int
    chunk_overlap: int
    em: float
    cem: float
    f1: float
    ms: float
    answer: str


@dataclass
class ConfigSummary:
    chunk_size: int
    chunk_overlap: int
    chunks: int
    avg_em: float
    avg_cem: float
    avg_f1: float
    avg_ms: float


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


def _containment_em(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    found = 0
    for token in ref_tokens:
        count = pred_counts.get(token, 0)
        if count > 0:
            found += 1
            pred_counts[token] = count - 1
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
    best_em = 0.0
    best_cem = 0.0
    best_f1 = 0.0
    for reference in references:
        best_em = max(best_em, _exact_match(prediction, reference))
        best_cem = max(best_cem, _containment_em(prediction, reference))
        best_f1 = max(best_f1, _f1(prediction, reference))
    return best_em, best_cem, best_f1


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_preview(text: str, max_chars: int = 180) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _load_questions(eval_set_path: Path, question_set_key: str, question_ids: list[str]) -> list[QuestionSpec]:
    payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
    raw_items = payload.get(question_set_key, [])
    if not isinstance(raw_items, list):
        raise ValueError(f"Question set '{question_set_key}' is not a list.")

    items_by_id = {
        str(item.get("id", "")).strip(): item
        for item in raw_items
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }

    selected: list[QuestionSpec] = []
    for question_id in question_ids:
        item = items_by_id.get(question_id)
        if item is None:
            raise ValueError(f"Question id '{question_id}' not found in '{question_set_key}'.")
        references = [str(text).strip() for text in item.get("references", []) if str(text).strip()]
        if not references:
            raise ValueError(f"Question id '{question_id}' has no references.")
        selected.append(
            QuestionSpec(
                question_id=question_id,
                question=str(item.get("question", "")).strip(),
                references=references,
                hypothesis=str(item.get("hypothesis", "")).strip(),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return selected


def _run_config(
    docs: list[Path],
    questions: list[QuestionSpec],
    chunk_size: int,
    chunk_overlap: int,
    retrieval_mode: str,
    ollama_model: str,
    retriever_k: int,
) -> tuple[ConfigSummary, list[QuestionRun]]:
    logger = configure_logging()
    settings = replace(
        load_settings(),
        ollama_model=ollama_model,
        retriever_k=max(1, retriever_k),
    )
    pipeline = RAGPipeline(settings=settings, logger=logger)
    uploads = [LocalUploadedFile(path) for path in docs]
    ingestion = pipeline.ingest_files(uploads, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    question_runs: list[QuestionRun] = []
    for question in questions:
        started = time.perf_counter()
        answer = pipeline.answer(
            question=question.question,
            chat_history=None,
            retrieval_mode=retrieval_mode,
            metadata_filters=None,
            enable_rerank=False,
            enable_self_rag=False,
            conversational=False,
            enable_query_rewrite=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        em, cem, f1 = _score_prediction(answer.answer, question.references)
        question_runs.append(
            QuestionRun(
                question_id=question.question_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                em=em * 100.0,
                cem=cem * 100.0,
                f1=f1 * 100.0,
                ms=elapsed_ms,
                answer=answer.answer,
            )
        )

    summary = ConfigSummary(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=ingestion.chunks,
        avg_em=_mean([item.em for item in question_runs]),
        avg_cem=_mean([item.cem for item in question_runs]),
        avg_f1=_mean([item.f1 for item in question_runs]),
        avg_ms=_mean([item.ms for item in question_runs]),
    )
    return summary, question_runs


def _build_report(
    generated_at: str,
    docs: list[Path],
    questions: list[QuestionSpec],
    config_summaries: list[ConfigSummary],
    question_runs: list[QuestionRun],
    retrieval_mode: str,
    ollama_model: str,
) -> str:
    lines: list[str] = []
    lines.append("# Chunk Benchmark Multi-Question Report")
    lines.append("")
    lines.append(f"- Generated at: {generated_at}")
    lines.append(f"- Model: `{ollama_model}`")
    lines.append(f"- Retrieval mode: `{retrieval_mode}`")
    lines.append(f"- Documents: {', '.join(doc.name for doc in docs)}")
    lines.append("")

    lines.append("## Question Set")
    for question in questions:
        lines.append(f"- **{question.question_id}**: {question.question}")
        if question.hypothesis:
            lines.append(f"  Expected chunk profile: {question.hypothesis}")
        if question.rationale:
            lines.append(f"  Why: {question.rationale}")
        lines.append(f"  Reference: {_safe_preview(question.references[0], 220)}")
    lines.append("")

    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| Chunk Size | Overlap | Chunks | Avg EM (%) | Avg CEM (%) | Avg F1 (%) | Avg ms |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(config_summaries, key=lambda item: (-item.avg_f1, -item.avg_cem, item.avg_ms)):
        lines.append(
            f"| {row.chunk_size} | {row.chunk_overlap} | {row.chunks} | "
            f"{row.avg_em:.1f} | {row.avg_cem:.1f} | {row.avg_f1:.1f} | {row.avg_ms:.0f} |"
        )
    lines.append("")

    best_overall = max(config_summaries, key=lambda item: (item.avg_f1, item.avg_cem, -item.avg_ms))
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"Best average configuration on this question set is **{best_overall.chunk_size}/{best_overall.chunk_overlap}** "
        f"with Avg F1 **{best_overall.avg_f1:.1f}%**, Avg CEM **{best_overall.avg_cem:.1f}%**, "
        f"and Avg latency **{best_overall.avg_ms:.0f} ms**."
    )
    lines.append("")

    lines.append("## Per-Question Breakdown")
    lines.append("")
    for question in questions:
        lines.append(f"### {question.question_id}")
        lines.append("")
        lines.append(question.question)
        lines.append("")
        lines.append("| Chunk Size | Overlap | EM (%) | CEM (%) | F1 (%) | ms |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        question_rows = [item for item in question_runs if item.question_id == question.question_id]
        question_rows.sort(key=lambda item: (-item.f1, -item.cem, item.ms))
        for row in question_rows:
            lines.append(
                f"| {row.chunk_size} | {row.chunk_overlap} | {row.em:.1f} | "
                f"{row.cem:.1f} | {row.f1:.1f} | {row.ms:.0f} |"
            )
        winner = question_rows[0]
        lines.append("")
        lines.append(
            f"Winner for **{question.question_id}**: **{winner.chunk_size}/{winner.chunk_overlap}** "
            f"(F1 {winner.f1:.1f}%, CEM {winner.cem:.1f}%)."
        )
        if question.hypothesis:
            lines.append("")
            lines.append(f"Expected profile: {question.hypothesis}")
        if question.rationale:
            lines.append(f"Reasoning: {question.rationale}")
        lines.append("")
        lines.append(f"Sample answer: {_safe_preview(winner.answer, 320)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple chunk configurations across multiple questions.")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=Path("data/real_docs/downloads"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/real_docs/pdf_eval_set.json"))
    parser.add_argument("--question-set-key", type=str, default="table2_questions")
    parser.add_argument("--question-ids", nargs="*", default=["Q1", "Q2", "Q3"])
    parser.add_argument("--chunk-sizes", nargs="*", type=int, default=[500, 1000, 1500, 2000])
    parser.add_argument("--chunk-overlaps", nargs="*", type=int, default=[50, 100, 200])
    parser.add_argument("--retrieval-mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:7b")
    parser.add_argument("--retriever-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("documentation/chunk_benchmark_multi_qwen2.5_7b.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    eval_set_path = args.eval_set.resolve()
    output_path = args.output.resolve()

    if not eval_set_path.exists():
        raise FileNotFoundError(f"Missing eval set file: {eval_set_path}")

    docs: list[Path]
    if args.pdf is not None:
        pdf_path = args.pdf.resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF file: {pdf_path}")
        docs = [pdf_path]
    else:
        docs_dir = args.docs_dir.resolve()
        if not docs_dir.exists():
            raise FileNotFoundError(f"Missing docs directory: {docs_dir}")
        docs = sorted(docs_dir.glob("*.pdf"))
        if not docs:
            raise ValueError(f"No PDF documents found in: {docs_dir}")

    questions = _load_questions(eval_set_path, args.question_set_key, args.question_ids)
    configs = [(size, overlap) for size in args.chunk_sizes for overlap in args.chunk_overlaps if overlap < size]
    if not configs:
        raise ValueError("No valid chunk configurations after filtering overlap < size.")

    all_summaries: list[ConfigSummary] = []
    all_question_runs: list[QuestionRun] = []

    total = len(configs)
    for index, (chunk_size, chunk_overlap) in enumerate(configs, start=1):
        print(
            f"[{index}/{total}] Benchmarking size={chunk_size}, overlap={chunk_overlap} "
            f"on {len(questions)} questions with model {args.ollama_model}"
        )
        summary, question_runs = _run_config(
            docs=docs,
            questions=questions,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_mode=args.retrieval_mode,
            ollama_model=args.ollama_model,
            retriever_k=args.retriever_k,
        )
        all_summaries.append(summary)
        all_question_runs.extend(question_runs)
        print(
            f"    Avg F1={summary.avg_f1:.1f}% | Avg CEM={summary.avg_cem:.1f}% | "
            f"Avg ms={summary.avg_ms:.0f} | chunks={summary.chunks}"
        )

    report = _build_report(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        docs=docs,
        questions=questions,
        config_summaries=all_summaries,
        question_runs=all_question_runs,
        retrieval_mode=args.retrieval_mode,
        ollama_model=args.ollama_model,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    best = max(all_summaries, key=lambda item: (item.avg_f1, item.avg_cem, -item.avg_ms))
    print("")
    print(
        f"Best overall: size={best.chunk_size}, overlap={best.chunk_overlap}, "
        f"Avg F1={best.avg_f1:.1f}%, Avg CEM={best.avg_cem:.1f}%, Avg ms={best.avg_ms:.0f}"
    )
    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
