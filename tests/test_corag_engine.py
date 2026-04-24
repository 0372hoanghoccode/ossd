from __future__ import annotations

import json

from langchain_core.documents import Document

from src.corag_engine import CoRAGStep, run_corag


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


class FakeRetriever:
    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[dict[str, object]] = []

    def __call__(self, query: str, step_k: int) -> list[Document]:
        self.call_count += 1
        self.calls.append({"query": query, "step_k": step_k})
        return [
            Document(
                page_content=f"content for {query}",
                metadata={
                    "source_name": "doc.md",
                    "chunk_id": self.call_count,
                    "start_index": self.call_count,
                    "page": self.call_count,
                },
            )
        ]


def _eval_payload(
    sufficient: bool,
    missing_parts: list[str],
    next_query: str | None,
    new_entities: dict[str, str] | None = None,
    reasoning: str = "ok",
) -> str:
    payload = {
        "sufficient": sufficient,
        "evidence_map": [
            {"part": part, "covered": part not in missing_parts, "value": None}
            for part in ["p1", "p2", "cluster", "manager", "deadline"]
        ],
        "missing_parts": missing_parts,
        "next_query": next_query,
        "new_entities": new_entities or {},
        "reasoning": reasoning,
    }
    return json.dumps(payload, ensure_ascii=False)


def test_terminates_at_max_steps_when_never_sufficient() -> None:
    fake_llm = FakeLLM("test")
    fake_llm.responses = [
        _eval_payload(False, ["p1"], "query step 2"),
        _eval_payload(False, ["p1"], "query step 3"),
        _eval_payload(False, ["p1"], "query step 4"),
        "final answer",
    ]
    fake_retriever = FakeRetriever()

    result = run_corag(
        question="main question",
        retrieve_fn=fake_retriever,
        llm_invoke_fn=lambda prompt: fake_llm.invoke(prompt),
        required_parts=["p1"],
        max_steps=3,
        step_k=2,
    )

    assert result.steps == 3
    assert len(result.chain) == 3
    assert result.sufficient is False
    assert result.answer == "final answer"


def test_stops_early_when_sufficient() -> None:
    fake_llm = FakeLLM("test")
    fake_llm.responses = [
        _eval_payload(False, ["p1"], "query step 2"),
        _eval_payload(True, [], None),
        "final early answer",
    ]
    fake_retriever = FakeRetriever()

    result = run_corag(
        question="another question",
        retrieve_fn=fake_retriever,
        llm_invoke_fn=lambda prompt: fake_llm.invoke(prompt),
        required_parts=["p1"],
        max_steps=4,
        step_k=2,
    )

    assert len(result.chain) == 2
    assert result.steps == 2
    assert result.chain[-1].sufficient is True
    assert fake_retriever.call_count == 2


def test_seen_queries_prevents_duplicate_retrieval() -> None:
    fake_llm = FakeLLM("test")
    fake_llm.responses = [
        _eval_payload(False, ["p1"], "  INITIAL   QUERY "),
        "final answer",
    ]
    fake_retriever = FakeRetriever()

    result = run_corag(
        question="Initial Query",
        retrieve_fn=fake_retriever,
        llm_invoke_fn=lambda prompt: fake_llm.invoke(prompt),
        required_parts=["p1"],
        max_steps=4,
        step_k=2,
    )

    assert len(result.chain) == 1
    assert fake_retriever.call_count == 1


def test_next_query_uses_discovered_entities() -> None:
    fake_llm = FakeLLM("test")
    fake_llm.responses = [
        _eval_payload(
            False,
            ["manager", "deadline"],
            "ai phu trach Cluster Gamma",
            new_entities={"cluster_most_incidents": "Cluster Gamma"},
            reasoning="found the cluster name",
        ),
        _eval_payload(
            True,
            [],
            None,
            new_entities={"manager": "Tran Van Duc", "deadline": "thang 9/2025"},
            reasoning="all parts covered",
        ),
        "entity grounded answer",
    ]

    issued_queries: list[str] = []

    def tracking_retrieve(query: str, step_k: int) -> list[Document]:
        issued_queries.append(query)
        return [
            Document(
                page_content=f"evidence for {query}",
                metadata={
                    "source_name": "trace.md",
                    "chunk_id": len(issued_queries),
                    "start_index": len(issued_queries),
                },
            )
        ]

    result = run_corag(
        question="Who manages the cluster with most incidents and what is their deadline?",
        retrieve_fn=tracking_retrieve,
        llm_invoke_fn=lambda prompt: fake_llm.invoke(prompt),
        required_parts=["cluster", "manager", "deadline"],
        max_steps=4,
        step_k=2,
    )

    assert len(issued_queries) >= 2
    assert "Cluster Gamma" in issued_queries[1]
    assert result.chain[0].discovered_entities["cluster_most_incidents"] == "Cluster Gamma"


def test_step_callback_called_for_each_step() -> None:
    fake_llm = FakeLLM("test")
    fake_llm.responses = [
        _eval_payload(False, ["p1"], "query step 2"),
        _eval_payload(True, [], None),
        "final answer",
    ]
    fake_retriever = FakeRetriever()

    callback_steps: list[CoRAGStep] = []

    def on_step(step: CoRAGStep) -> None:
        callback_steps.append(step)

    result = run_corag(
        question="callback question",
        retrieve_fn=fake_retriever,
        llm_invoke_fn=lambda prompt: fake_llm.invoke(prompt),
        required_parts=["p1"],
        max_steps=4,
        step_k=2,
        step_callback=on_step,
    )

    assert len(callback_steps) == len(result.chain)
    assert callback_steps[0].step == 1
    assert callback_steps[1].step == 2
    assert callback_steps[0].retrieved_count >= 0


def test_empty_retrieval_does_not_crash() -> None:
    fake_llm = FakeLLM("test")
    fake_llm.responses = [
        _eval_payload(False, ["p1"], None),
        "final answer with no docs",
    ]

    def empty_retrieve(_query: str, _step_k: int) -> list[Document]:
        return []

    result = run_corag(
        question="question with empty retrieval",
        retrieve_fn=empty_retrieve,
        llm_invoke_fn=lambda prompt: fake_llm.invoke(prompt),
        required_parts=["p1"],
        max_steps=3,
        step_k=2,
    )

    assert result.answer == "final answer with no docs"
    assert len(result.chain) == 1
    assert result.chain[0].retrieved_count == 0
    assert result.total_docs == 0
