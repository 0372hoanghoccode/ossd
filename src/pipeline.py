from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import docx
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from src.config import Settings
from src.prompts import (
    build_conversational_prompt,
    build_prompt,
    build_query_rewrite_prompt,
    build_self_reflection_prompt,
    format_chat_history,
    is_vietnamese,
    parse_reflection,
)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


_EMBEDDER_CACHE: dict[str, Any] = {}


@dataclass
class IngestionResult:
    source_paths: list[Path]
    chunks: int
    seconds: float
    chunk_size: int
    chunk_overlap: int
    files_indexed: int


@dataclass
class ChunkExperimentResult:
    chunk_size: int
    chunk_overlap: int
    chunks: int
    split_seconds: float
    accuracy_recall_at_k: float
    benchmark_queries: int


@dataclass
class AnswerResult:
    answer: str
    sources: list[Document]
    retrieval_seconds: float
    generation_seconds: float
    mode: str
    confidence: float
    rationale: str
    used_query: str


class RAGPipeline:
    def __init__(self, settings: Settings, logger: Any):
        self.settings = settings
        self.logger = logger
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.faiss_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = None
        self._llm = OllamaLLM(model=self.settings.ollama_model)
        self._vector_store: FAISS | None = None
        self._retriever = None
        self._all_documents: list[Document] = []
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []
        self._cross_encoder: Any | None = None

    @property
    def is_ready(self) -> bool:
        return self._retriever is not None

    @property
    def document_count(self) -> int:
        return len(self._all_documents)

    def _load_pdf(self, file_path: Path) -> list[Document]:
        loader = PDFPlumberLoader(str(file_path))
        return loader.load()

    def _load_docx(self, file_path: Path) -> list[Document]:
        parsed = docx.Document(str(file_path))
        blocks = [paragraph.text.strip() for paragraph in parsed.paragraphs if paragraph.text.strip()]

        for table in parsed.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))

        documents: list[Document] = []
        for idx, block_text in enumerate(blocks, start=1):
            documents.append(
                Document(
                    page_content=block_text,
                    metadata={"source": str(file_path), "page": idx},
                )
            )
        return documents

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _build_benchmark_queries(
        self,
        split_docs: list[Document],
        max_queries: int = 6,
    ) -> list[tuple[str, int]]:
        if not split_docs:
            return []

        if len(split_docs) <= max_queries:
            sample_indices = list(range(len(split_docs)))
        else:
            step = (len(split_docs) - 1) / max(max_queries - 1, 1)
            sample_indices = sorted({int(round(i * step)) for i in range(max_queries)})

        queries: list[tuple[str, int]] = []
        for idx in sample_indices:
            tokens = self._tokenize(split_docs[idx].page_content)
            if len(tokens) < 6:
                continue
            start = max((len(tokens) // 3) - 4, 0)
            query_tokens = tokens[start : start + 8]
            if not query_tokens:
                continue
            queries.append((" ".join(query_tokens), idx))
        return queries

    def _estimate_chunk_recall_at_k(
        self,
        split_docs: list[Document],
        top_k: int,
    ) -> tuple[float, int]:
        if not split_docs:
            return 0.0, 0

        corpus = [self._tokenize(doc.page_content) for doc in split_docs]
        if not corpus:
            return 0.0, 0

        bm25 = BM25Okapi(corpus)
        benchmark_queries = self._build_benchmark_queries(split_docs)
        if not benchmark_queries:
            return 0.0, 0

        effective_k = max(1, top_k)
        hits = 0
        evaluated = 0

        for query, target_idx in benchmark_queries:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                continue
            scores = bm25.get_scores(query_tokens)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = ranked_indices[:effective_k]
            evaluated += 1
            if target_idx in top_indices:
                hits += 1

        if evaluated == 0:
            return 0.0, 0
        return hits / evaluated, evaluated

    def _ensure_cross_encoder(self) -> Any | None:
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.settings.rerank_model)
        except Exception as exc:
            self.logger.warning("Cross-encoder unavailable, fallback to retrieval ranking: %s", exc)
            self._cross_encoder = None
        return self._cross_encoder

    def _ensure_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        try:
            model_key = self.settings.embedding_model
            if model_key not in _EMBEDDER_CACHE:
                _EMBEDDER_CACHE[model_key] = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            self._embedder = _EMBEDDER_CACHE[model_key]
            return self._embedder
        except ModuleNotFoundError as exc:
            if "sentence_transformers" in str(exc):
                raise ImportError(
                    "Embedding dependency is missing. Install with: pip install sentence-transformers"
                ) from exc
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize embeddings: {exc}") from exc

    def _rebuild_indices(self, new_docs: list[Document] | None = None) -> None:
        if not self._all_documents:
            self._vector_store = None
            self._retriever = None
            self._bm25_index = None
            self._bm25_corpus = []
            return

        embedder = self._ensure_embedder()
        if self._vector_store is not None and new_docs:
            self._vector_store.add_documents(new_docs)
        else:
            self._vector_store = FAISS.from_documents(self._all_documents, embedder)

        self._vector_store.save_local(str(self.settings.faiss_dir))
        self._retriever = self._vector_store.as_retriever(
            search_type=self.settings.retriever_search_type,
            search_kwargs={"k": self.settings.retriever_k},
        )

        self._bm25_corpus = [self._tokenize(doc.page_content) for doc in self._all_documents]
        self._bm25_index = BM25Okapi(self._bm25_corpus) if self._bm25_corpus else None

    def _apply_metadata_filters(
        self,
        docs: list[Document],
        metadata_filters: dict[str, list[str]] | None,
    ) -> list[Document]:
        if not metadata_filters:
            return docs

        source_filter = set(metadata_filters.get("source_name", []))
        type_filter = set(metadata_filters.get("doc_type", []))

        filtered: list[Document] = []
        for doc in docs:
            source_name = str(doc.metadata.get("source_name", ""))
            doc_type = str(doc.metadata.get("doc_type", ""))
            source_ok = not source_filter or source_name in source_filter
            type_ok = not type_filter or doc_type in type_filter
            if source_ok and type_ok:
                filtered.append(doc)
        return filtered

    def _vector_retrieve(
        self,
        query: str,
        fetch_k: int,
        metadata_filters: dict[str, list[str]] | None,
    ) -> list[Document]:
        if not self._vector_store:
            return []

        pairs = self._vector_store.similarity_search_with_relevance_scores(query, k=max(fetch_k, 1))
        docs: list[Document] = []
        for doc, score in pairs:
            doc.metadata["retrieval_score"] = float(score)
            docs.append(doc)

        filtered = self._apply_metadata_filters(docs, metadata_filters)
        return filtered[: self.settings.retriever_k]

    def _keyword_retrieve(
        self,
        query: str,
        fetch_k: int,
        metadata_filters: dict[str, list[str]] | None,
    ) -> list[Document]:
        if not self._bm25_index:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)
        indexed_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:fetch_k]
        docs: list[Document] = []
        for idx, score in indexed_scores:
            doc = self._all_documents[idx]
            copy_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            copy_doc.metadata["retrieval_score"] = float(score)
            docs.append(copy_doc)

        filtered = self._apply_metadata_filters(docs, metadata_filters)
        return filtered[: self.settings.retriever_k]

    def _hybrid_retrieve(
        self,
        query: str,
        fetch_k: int,
        metadata_filters: dict[str, list[str]] | None,
    ) -> list[Document]:
        vector_docs = self._vector_retrieve(query, fetch_k, metadata_filters)
        keyword_docs = self._keyword_retrieve(query, fetch_k, metadata_filters)

        merged: dict[str, tuple[Document, float]] = {}

        def doc_key(doc: Document) -> str:
            return (
                f"{doc.metadata.get('source_name', '')}|"
                f"{doc.metadata.get('chunk_id', '')}|"
                f"{doc.metadata.get('start_index', '')}"
            )

        for rank, doc in enumerate(vector_docs):
            score = self.settings.hybrid_alpha * (1.0 / (rank + 1))
            merged[doc_key(doc)] = (doc, score)

        for rank, doc in enumerate(keyword_docs):
            key = doc_key(doc)
            score = (1.0 - self.settings.hybrid_alpha) * (1.0 / (rank + 1))
            if key in merged:
                current_doc, current_score = merged[key]
                merged[key] = (current_doc, current_score + score)
            else:
                merged[key] = (doc, score)

        reranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)[: self.settings.retriever_k]
        docs = [doc for doc, score in reranked]
        for doc, score in reranked:
            doc.metadata["retrieval_score"] = float(score)
        return docs

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        cross_encoder = self._ensure_cross_encoder()
        if cross_encoder is None:
            return docs

        top_n = min(max(self.settings.rerank_top_n, 1), len(docs))
        candidates = docs[:top_n]
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)

        reranked_docs = [doc for doc, _ in ranked]
        for doc, score in ranked:
            doc.metadata["rerank_score"] = float(score)
        return reranked_docs

    def _detect_file_type(self, uploaded_file: Any) -> tuple[str, str]:
        filename = (uploaded_file.name or "").lower()
        if filename.endswith(".pdf"):
            return ".pdf", "pdf"
        if filename.endswith(".docx"):
            return ".docx", "docx"
        raise ValueError("Only .pdf and .docx files are supported.")

    def _load_uploaded_documents(self, uploaded_file: Any) -> tuple[Path, list[Document]]:
        suffix, file_type = self._detect_file_type(uploaded_file)

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        if file_type == "pdf":
            docs = self._load_pdf(temp_path)
        else:
            docs = self._load_docx(temp_path)

        return temp_path, docs

    def ingest_file(
        self,
        uploaded_file: Any,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        return self.ingest_files([uploaded_file], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_files(
        self,
        uploaded_files: list[Any],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        started = time.perf_counter()

        effective_chunk_size = chunk_size if chunk_size is not None else self.settings.chunk_size
        effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else self.settings.chunk_overlap

        all_new_docs: list[Document] = []
        source_paths: list[Path] = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            add_start_index=True,
        )

        for uploaded_file in uploaded_files:
            temp_path, docs = self._load_uploaded_documents(uploaded_file)
            source_paths.append(temp_path)

            split_docs = splitter.split_documents(docs)
            source_name = uploaded_file.name or temp_path.name
            uploaded_at = datetime.now().isoformat(timespec="seconds")
            doc_type = source_name.split(".")[-1].lower() if "." in source_name else "unknown"

            for idx, doc in enumerate(split_docs, start=1):
                doc.metadata["chunk_id"] = idx
                doc.metadata["source_name"] = source_name
                doc.metadata["uploaded_at"] = uploaded_at
                doc.metadata["doc_type"] = doc_type
                doc.metadata.setdefault("start_index", -1)
                all_new_docs.append(doc)

        self._all_documents.extend(all_new_docs)
        self._rebuild_indices(new_docs=all_new_docs)

        elapsed = time.perf_counter() - started
        self.logger.info(
            "Indexed %s chunks from %s file(s) in %.2fs",
            len(all_new_docs),
            len(uploaded_files),
            elapsed,
        )

        return IngestionResult(
            source_paths=source_paths,
            chunks=len(all_new_docs),
            seconds=elapsed,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            files_indexed=len(uploaded_files),
        )

    def benchmark_chunk_strategies(
        self,
        uploaded_file: Any,
        chunk_sizes: list[int],
        chunk_overlaps: list[int],
    ) -> list[ChunkExperimentResult]:
        _, docs = self._load_uploaded_documents(uploaded_file)

        results: list[ChunkExperimentResult] = []
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                started = time.perf_counter()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    add_start_index=True,
                )
                split_docs = splitter.split_documents(docs)
                elapsed = time.perf_counter() - started
                accuracy_recall_at_k, benchmark_queries = self._estimate_chunk_recall_at_k(
                    split_docs,
                    top_k=self.settings.retriever_k,
                )
                results.append(
                    ChunkExperimentResult(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunks=len(split_docs),
                        split_seconds=elapsed,
                        accuracy_recall_at_k=accuracy_recall_at_k,
                        benchmark_queries=benchmark_queries,
                    )
                )

        return results

    def clear_vector_store(self) -> None:
        self._vector_store = None
        self._retriever = None
        self._all_documents = []
        self._bm25_index = None
        self._bm25_corpus = []
        if self.settings.faiss_dir.exists():
            shutil.rmtree(self.settings.faiss_dir)
        self.settings.faiss_dir.mkdir(parents=True, exist_ok=True)

    def list_available_sources(self) -> dict[str, list[str]]:
        source_names = sorted({str(doc.metadata.get("source_name", "")) for doc in self._all_documents if doc.metadata.get("source_name")})
        doc_types = sorted({str(doc.metadata.get("doc_type", "")) for doc in self._all_documents if doc.metadata.get("doc_type")})
        return {"source_name": source_names, "doc_type": doc_types}

    def _retrieve(
        self,
        query: str,
        mode: str,
        metadata_filters: dict[str, list[str]] | None,
    ) -> list[Document]:
        fetch_k = max(self.settings.bm25_fetch_k, self.settings.retriever_k)
        if mode == "keyword":
            return self._keyword_retrieve(query, fetch_k, metadata_filters)
        if mode == "hybrid":
            return self._hybrid_retrieve(query, fetch_k, metadata_filters)
        return self._vector_retrieve(query, fetch_k, metadata_filters)

    def answer(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        retrieval_mode: str = "vector",
        metadata_filters: dict[str, list[str]] | None = None,
        enable_rerank: bool = False,
        enable_self_rag: bool = False,
        conversational: bool = True,
    ) -> AnswerResult:
        if not self._all_documents:
            raise ValueError("Retriever is not initialized. Upload and process a document first.")

        chat_history = chat_history or []
        history_text = format_chat_history(chat_history, self.settings.conversational_memory_turns)

        used_query = question
        if conversational and chat_history:
            try:
                rewrite_prompt = build_query_rewrite_prompt(question=question, chat_history=history_text)
                rewritten = str(self._llm.invoke(rewrite_prompt)).strip()
                if rewritten:
                    used_query = rewritten
            except Exception as exc:
                self.logger.warning("Query rewrite failed, fallback to original query: %s", exc)

        retrieve_started = time.perf_counter()
        docs = self._retrieve(used_query, retrieval_mode, metadata_filters)
        if enable_rerank:
            docs = self._rerank(used_query, docs)
        retrieval_elapsed = time.perf_counter() - retrieve_started

        if not docs:
            fallback_answer = (
                "Không tìm thấy ngữ cảnh phù hợp với bộ lọc hiện tại."
                if is_vietnamese(question)
                else "No relevant context found for current filters."
            )
            return AnswerResult(
                answer=fallback_answer,
                sources=[],
                retrieval_seconds=retrieval_elapsed,
                generation_seconds=0.0,
                mode=retrieval_mode,
                confidence=0.0,
                rationale="No context after retrieval/filtering.",
                used_query=used_query,
            )

        context = "\n\n".join(doc.page_content for doc in docs)
        if conversational:
            prompt = build_conversational_prompt(context=context, question=question, chat_history=history_text)
        else:
            prompt = build_prompt(context=context, question=question)

        generate_started = time.perf_counter()
        answer = self._llm.invoke(prompt)
        generation_elapsed = time.perf_counter() - generate_started

        final_answer = str(answer).strip()
        confidence: float | None = None
        rationale = ""

        if enable_self_rag:
            if not context.strip():
                confidence = 0.0
                rationale = "No context to reflect on."
            else:
                try:
                    reflection_prompt = build_self_reflection_prompt(question=question, answer=final_answer, context=context)
                    reflection_text = str(self._llm.invoke(reflection_prompt))
                    confidence, needs_rewrite, rewrite_query, rationale = parse_reflection(reflection_text)
                    rationale = rationale or "Self-RAG reflection completed."

                    if needs_rewrite and rewrite_query and rewrite_query.lower() != used_query.lower():
                        used_query = rewrite_query
                        second_docs = self._retrieve(used_query, retrieval_mode, metadata_filters)
                        if enable_rerank:
                            second_docs = self._rerank(used_query, second_docs)
                        if second_docs:
                            second_context = "\n\n".join(doc.page_content for doc in second_docs)
                            second_prompt = (
                                build_conversational_prompt(second_context, question, history_text)
                                if conversational
                                else build_prompt(second_context, question)
                            )
                            second_answer = str(self._llm.invoke(second_prompt)).strip()
                            second_reflection = str(
                                self._llm.invoke(
                                    build_self_reflection_prompt(
                                        question=question,
                                        answer=second_answer,
                                        context=second_context,
                                    )
                                )
                            )
                            second_confidence, _, _, second_rationale = parse_reflection(second_reflection)
                            if second_confidence >= (confidence if confidence is not None else 0.0):
                                final_answer = second_answer
                                docs = second_docs
                                confidence = second_confidence
                                rationale = second_rationale or rationale
                except Exception as exc:
                    self.logger.warning("Self-RAG reflection failed: %s", exc)
                    confidence = 0.6
                    rationale = "Base RAG answer generated."
        else:
            confidence = 0.6
            rationale = "Base RAG answer generated."

        if confidence is None:
            confidence = 0.6
        if not rationale:
            rationale = "Base RAG answer generated."

        self.logger.info(
            "Answered query in %.2fs (retrieve=%.2fs, generate=%.2fs)",
            retrieval_elapsed + generation_elapsed,
            retrieval_elapsed,
            generation_elapsed,
        )

        return AnswerResult(
            answer=final_answer,
            sources=docs,
            retrieval_seconds=retrieval_elapsed,
            generation_seconds=generation_elapsed,
            mode=retrieval_mode,
            confidence=confidence,
            rationale=rationale,
            used_query=used_query,
        )
