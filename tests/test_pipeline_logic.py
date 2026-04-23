from __future__ import annotations

import logging
from pathlib import Path

import docx
import pytest
from langchain_core.documents import Document

import src.pipeline as pipeline_module
from src.config import Settings
from src.pipeline import RAGPipeline


class FakeLLM:
    def __init__(self, model: str):
        self.model = model
        self.responses: list[object] = []
        self.prompts: list[str] = []

    def invoke(self, prompt: object) -> str:
        self.prompts.append(str(prompt))
        if self.responses:
            next_response = self.responses.pop(0)
            if isinstance(next_response, Exception):
                raise next_response
            return str(next_response)
        return "mock-response"


class FakeUploadedFile:
    def __init__(self, name: str, payload: bytes = b"dummy"):
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        return self._payload


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "ollama_model": "test-llm",
        "embedding_model": "test-embed",
        "rerank_model": "test-rerank",
        "chunk_size": 120,
        "chunk_overlap": 20,
        "retriever_k": 3,
        "retriever_search_type": "similarity",
        "hybrid_alpha": 0.6,
        "bm25_fetch_k": 5,
        "rerank_top_n": 2,
        "conversational_memory_turns": 4,
        "data_dir": tmp_path / "data",
        "faiss_dir": tmp_path / "faiss",
    }
    base.update(overrides)
    return Settings(**base)


def make_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **settings_overrides: object) -> RAGPipeline:
    monkeypatch.setattr(pipeline_module, "OllamaLLM", FakeLLM)
    settings = make_settings(tmp_path, **settings_overrides)
    logger = logging.getLogger("test.pipeline")
    logger.setLevel(logging.CRITICAL)
    return RAGPipeline(settings=settings, logger=logger)


def test_load_docx_extracts_paragraphs_and_tables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = make_pipeline(monkeypatch, tmp_path)

    sample_docx = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_paragraph("Executive summary")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Control"
    table.cell(0, 1).text = "Status"
    table.cell(1, 0).text = "AC-1"
    table.cell(1, 1).text = "Implemented"
    doc.save(sample_docx)

    loaded = pipeline._load_docx(sample_docx)

    assert len(loaded) == 1
    content = loaded[0].page_content
    assert "Executive summary" in content
    assert "Control | Status" in content
    assert "AC-1 | Implemented" in content


def test_ingest_files_populates_required_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = make_pipeline(monkeypatch, tmp_path)

    def fake_loader(_uploaded: object) -> tuple[Path, list[Document]]:
        text = " ".join(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"] * 15)
        return tmp_path / "fake.docx", [Document(page_content=text, metadata={"page": 1})]

    monkeypatch.setattr(pipeline, "_load_uploaded_documents", fake_loader)
    monkeypatch.setattr(pipeline, "_rebuild_indices", lambda: None)

    result = pipeline.ingest_files([FakeUploadedFile(name="audit.docx")], chunk_size=80, chunk_overlap=10)

    assert result.files_indexed == 1
    assert pipeline.document_count > 0
    first = pipeline._all_documents[0]
    assert first.metadata["source_name"] == "audit.docx"
    assert first.metadata["doc_type"] == "docx"
    assert first.metadata["chunk_id"] >= 1
    assert "uploaded_at" in first.metadata
    assert "start_index" in first.metadata


def test_benchmark_chunk_strategies_includes_accuracy_metric(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = make_pipeline(monkeypatch, tmp_path)

    long_text = " ".join(
        [
            "incident",
            "response",
            "containment",
            "eradication",
            "recovery",
            "lessons",
            "controls",
            "risk",
            "mitigation",
            "evidence",
        ]
        * 80
    )

    def fake_loader(_uploaded: object) -> tuple[Path, list[Document]]:
        return tmp_path / "fake.pdf", [Document(page_content=long_text, metadata={"page": 1})]

    monkeypatch.setattr(pipeline, "_load_uploaded_documents", fake_loader)

    results = pipeline.benchmark_chunk_strategies(
        uploaded_file=FakeUploadedFile(name="nist.pdf"),
        chunk_sizes=[100],
        chunk_overlaps=[20],
    )

    assert len(results) == 1
    row = results[0]
    assert 0.0 <= row.accuracy_recall_at_k <= 1.0
    assert row.benchmark_queries > 0


def test_rerank_returns_only_reranked_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = make_pipeline(monkeypatch, tmp_path, rerank_top_n=2)

    class FakeCrossEncoder:
        @staticmethod
        def predict(_pairs: list[tuple[str, str]]) -> list[float]:
            return [0.2, 0.9]

    monkeypatch.setattr(pipeline, "_ensure_cross_encoder", lambda: FakeCrossEncoder())

    docs = [
        Document(page_content="doc-1", metadata={}),
        Document(page_content="doc-2", metadata={}),
        Document(page_content="doc-3", metadata={}),
    ]

    reranked = pipeline._rerank("risk controls", docs)

    assert len(reranked) == 2
    assert reranked[0].page_content == "doc-2"
    assert reranked[1].page_content == "doc-1"
    assert reranked[0].metadata["rerank_score"] == 0.9


def test_rerank_with_non_positive_topn_still_reranks_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = make_pipeline(monkeypatch, tmp_path, rerank_top_n=0)

    class FakeCrossEncoder:
        @staticmethod
        def predict(_pairs: list[tuple[str, str]]) -> list[float]:
            return [0.7]

    monkeypatch.setattr(pipeline, "_ensure_cross_encoder", lambda: FakeCrossEncoder())

    docs = [Document(page_content="only-doc", metadata={})]
    reranked = pipeline._rerank("query", docs)

    assert len(reranked) == 1
    assert reranked[0].metadata["rerank_score"] == 0.7


def test_answer_self_rag_confidence_comes_from_reflection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = make_pipeline(monkeypatch, tmp_path)
    pipeline._all_documents = [Document(page_content="seed", metadata={})]

    retrieved_docs = [Document(page_content="retrieved context", metadata={"source_name": "doc.pdf", "chunk_id": 1})]
    monkeypatch.setattr(pipeline, "_retrieve", lambda *_args, **_kwargs: retrieved_docs)

    llm = pipeline._llm
    assert isinstance(llm, FakeLLM)
    llm.responses = [
        "Base answer",
        "CONFIDENCE: 0.87\nNEEDS_REWRITE: no\nREWRITE_QUERY:\nRATIONALE: reflection ok",
    ]

    result = pipeline.answer(
        question="What is the control objective?",
        conversational=False,
        enable_self_rag=True,
    )

    assert result.answer == "Base answer"
    assert result.confidence == 0.87
    assert result.rationale == "reflection ok"


def test_answer_self_rag_reflection_failure_has_clear_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = make_pipeline(monkeypatch, tmp_path)
    pipeline._all_documents = [Document(page_content="seed", metadata={})]

    retrieved_docs = [Document(page_content="retrieved context", metadata={"source_name": "doc.pdf", "chunk_id": 1})]
    monkeypatch.setattr(pipeline, "_retrieve", lambda *_args, **_kwargs: retrieved_docs)

    llm = pipeline._llm
    assert isinstance(llm, FakeLLM)
    llm.responses = ["Base answer", RuntimeError("reflection offline")]

    result = pipeline.answer(
        question="What is the control objective?",
        conversational=False,
        enable_self_rag=True,
    )

    assert result.answer == "Base answer"
    assert result.confidence == 0.6
    assert result.rationale == "Base RAG answer generated."
