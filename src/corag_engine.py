from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document

from src.prompts import (
    build_corag_evaluator_prompt,
    build_corag_final_answer_prompt,
    is_vietnamese,
)


@dataclass
class CoRAGStep:
    step: int
    query: str
    retrieved_count: int
    sufficient: bool
    missing_parts: list[str]
    next_query: str | None
    discovered_entities: dict[str, str]
    reasoning: str


@dataclass
class CoRAGResult:
    answer: str
    chain: list[CoRAGStep]
    all_docs: list[Document]
    total_docs: int
    steps: int
    sufficient: bool


def _doc_key(doc: Document) -> str:
    """Stable dedup key from metadata, fallback to content hash."""
    source = str(doc.metadata.get("source_name", "")).strip()
    chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
    start = str(doc.metadata.get("start_index", "")).strip()
    if source and chunk_id:
        return f"{source}|{chunk_id}|{start}"
    return hashlib.md5(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_query(query: str) -> str:
    """Normalize for seen_queries dedup."""
    return re.sub(r"\s+", " ", query.lower().strip())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _parse_json_robust(text: str) -> dict:
    """
    1. Strip ```json ... ``` fences if present
    2. Extract first {...} object if needed
    3. json.loads()
    4. Return {} on any failure
    """
    if not text:
        return {}

    candidate = str(text).strip()
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    extracted = _extract_first_json_object(candidate)
    if not extracted:
        return {}

    try:
        data = json.loads(extracted)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_context_string(
    docs: list[Document],
    question: str,
    top_n: int = 8,
) -> str:
    """
    Rerank docs by lexical overlap with question plus simple recency.
    """
    if not docs:
        return ""

    q_terms = set(_tokenize(question))
    max_step = max(int(doc.metadata.get("__corag_step", 0) or 0) for doc in docs) if docs else 0
    max_step = max(max_step, 1)

    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(docs):
        d_terms = set(_tokenize(doc.page_content))
        overlap = len(q_terms & d_terms)
        lexical = overlap / max(len(q_terms), 1)

        step_num = int(doc.metadata.get("__corag_step", 0) or 0)
        recency = step_num / max_step

        score = (0.8 * lexical) + (0.2 * recency)
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [doc for _, _, doc in scored[: max(top_n, 1)]]

    chunks: list[str] = []
    for idx, doc in enumerate(selected, start=1):
        source = str(doc.metadata.get("source_name", doc.metadata.get("source", "unknown")))
        page = str(doc.metadata.get("page", "?"))
        chunk = str(doc.metadata.get("chunk_id", "?"))
        chunks.append(
            f"[Chunk {idx} | source={source} | page={page} | chunk={chunk}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(chunks)


def _safe_required_parts(required_parts: list[str], question: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in required_parts:
        text = str(part).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    if cleaned:
        return cleaned
    fallback = question.strip() or "main_answer"
    return [fallback]


def _evaluate(
    question: str,
    required_parts: list[str],
    context: str,
    discovered_entities: dict[str, str],
    llm_invoke_fn: Callable[[str], str],
) -> dict:
    prompt = build_corag_evaluator_prompt(
        question=question,
        required_parts=required_parts,
        context=context,
        discovered_entities=discovered_entities,
    )

    try:
        raw = str(llm_invoke_fn(prompt)).strip()
    except Exception:
        raw = ""

    parsed = _parse_json_robust(raw)
    if not parsed:
        return {
            "sufficient": False,
            "missing_parts": required_parts,
            "next_query": None,
            "new_entities": {},
            "reasoning": "Evaluator output is malformed.",
        }

    evidence_map = parsed.get("evidence_map", [])
    if not isinstance(evidence_map, list):
        evidence_map = []

    missing_parts = parsed.get("missing_parts", [])
    if not isinstance(missing_parts, list):
        missing_parts = []

    required_lookup = {part.lower(): part for part in required_parts}
    normalized_missing: list[str] = []
    for part in missing_parts:
        if not isinstance(part, str):
            continue
        key = part.strip().lower()
        if key in required_lookup and required_lookup[key] not in normalized_missing:
            normalized_missing.append(required_lookup[key])

    if not normalized_missing and evidence_map:
        for row in evidence_map:
            if not isinstance(row, dict):
                continue
            part_name = str(row.get("part", "")).strip().lower()
            covered = bool(row.get("covered", False))
            if part_name in required_lookup and not covered:
                candidate = required_lookup[part_name]
                if candidate not in normalized_missing:
                    normalized_missing.append(candidate)

    if not normalized_missing and not bool(parsed.get("sufficient", False)):
        normalized_missing = required_parts

    new_entities = parsed.get("new_entities", {})
    if not isinstance(new_entities, dict):
        new_entities = {}
    clean_entities: dict[str, str] = {}
    for key, value in new_entities.items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            clean_entities[k] = v

    if not clean_entities and evidence_map:
        for row in evidence_map:
            if not isinstance(row, dict):
                continue
            part_name = str(row.get("part", "")).strip()
            covered = bool(row.get("covered", False))
            value = row.get("value")
            if covered and part_name and isinstance(value, str) and value.strip():
                clean_entities[part_name] = value.strip()

    next_query = parsed.get("next_query")
    if not isinstance(next_query, str) or not next_query.strip():
        next_query = None
    else:
        next_query = next_query.strip()

    sufficient = bool(parsed.get("sufficient", False))
    if normalized_missing:
        sufficient = False

    if next_query and normalized_missing:
        known_entities = [str(value).strip() for value in discovered_entities.values()]
        known_entities.extend(str(value).strip() for value in clean_entities.values())
        known_entities = [value for value in known_entities if value]
        if known_entities:
            lowered_query = next_query.lower()
            has_grounding = any(entity.lower() in lowered_query for entity in known_entities)
            if not has_grounding:
                next_query = None

    reasoning = str(parsed.get("reasoning", "")).strip() or "No evaluator reasoning provided."

    return {
        "sufficient": sufficient,
        "missing_parts": normalized_missing,
        "next_query": next_query,
        "new_entities": clean_entities,
        "reasoning": reasoning,
    }


def run_corag(
    question: str,
    retrieve_fn: Callable[[str, int], list[Document]],
    llm_invoke_fn: Callable[[str], str],
    required_parts: list[str],
    max_steps: int = 4,
    step_k: int = 3,
    step_callback: Callable[[CoRAGStep], None] | None = None,
) -> CoRAGResult:
    safe_question = question.strip() or question
    safe_parts = _safe_required_parts(required_parts, safe_question)

    discovered_entities: dict[str, str] = {}
    seen_queries: set[str] = {_normalize_query(safe_question)}
    seen_doc_keys: set[str] = set()
    context_pool: list[Document] = []
    chain: list[CoRAGStep] = []

    current_query = safe_question
    effective_steps = max(1, int(max_steps))
    effective_k = max(1, int(step_k))

    for step in range(1, effective_steps + 1):
        try:
            retrieved_docs = retrieve_fn(current_query, effective_k)
        except Exception:
            retrieved_docs = []

        unique_new: list[Document] = []
        for doc in retrieved_docs:
            key = _doc_key(doc)
            if key in seen_doc_keys:
                continue
            seen_doc_keys.add(key)
            copy_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            copy_doc.metadata["__corag_step"] = step
            unique_new.append(copy_doc)

        context_pool.extend(unique_new)
        context_text = _build_context_string(context_pool, safe_question, top_n=8)

        eval_result = _evaluate(
            question=safe_question,
            required_parts=safe_parts,
            context=context_text,
            discovered_entities=discovered_entities,
            llm_invoke_fn=llm_invoke_fn,
        )

        discovered_entities.update(eval_result.get("new_entities", {}))

        step_data = CoRAGStep(
            step=step,
            query=current_query,
            retrieved_count=len(unique_new),
            sufficient=bool(eval_result.get("sufficient", False)),
            missing_parts=list(eval_result.get("missing_parts", [])),
            next_query=eval_result.get("next_query"),
            discovered_entities=dict(discovered_entities),
            reasoning=str(eval_result.get("reasoning", "")).strip(),
        )
        chain.append(step_data)

        if step_callback is not None:
            try:
                step_callback(step_data)
            except Exception:
                pass

        if step_data.sufficient:
            break

        next_query = (step_data.next_query or "").strip()
        if not next_query:
            break

        normalized_next = _normalize_query(next_query)
        if normalized_next in seen_queries:
            break

        seen_queries.add(normalized_next)
        current_query = next_query

    final_context = _build_context_string(context_pool, safe_question, top_n=10)
    language = "vi" if is_vietnamese(safe_question) else "en"
    final_prompt = build_corag_final_answer_prompt(
        question=safe_question,
        context=final_context,
        language=language,
    )

    try:
        answer = str(llm_invoke_fn(final_prompt)).strip()
    except Exception:
        answer = ""

    if not answer:
        answer = (
            "Không thể tạo câu trả lời từ ngữ cảnh hiện có."
            if language == "vi"
            else "Unable to produce an answer from the available context."
        )

    sufficient = chain[-1].sufficient if chain else False
    return CoRAGResult(
        answer=answer,
        chain=chain,
        all_docs=context_pool,
        total_docs=len(context_pool),
        steps=len(chain),
        sufficient=sufficient,
    )
