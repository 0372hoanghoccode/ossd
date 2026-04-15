from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_model: str
    embedding_model: str
    rerank_model: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    retriever_search_type: str
    hybrid_alpha: float
    bm25_fetch_k: int
    rerank_top_n: int
    conversational_memory_turns: int
    data_dir: Path
    faiss_dir: Path


def load_settings() -> Settings:
    load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    faiss_dir = Path(os.getenv("FAISS_DIR", str(data_dir / "faiss_index")))

    return Settings(
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        retriever_k=int(os.getenv("RETRIEVER_K", "3")),
        retriever_search_type=os.getenv("RETRIEVER_SEARCH_TYPE", "similarity"),
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.6")),
        bm25_fetch_k=int(os.getenv("BM25_FETCH_K", "8")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
        conversational_memory_turns=int(os.getenv("CONVERSATIONAL_MEMORY_TURNS", "4")),
        data_dir=data_dir,
        faiss_dir=faiss_dir,
    )
