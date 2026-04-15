from src.prompts import (
    build_conversational_prompt,
    build_query_rewrite_prompt,
    build_self_reflection_prompt,
    format_chat_history,
    parse_reflection,
)


def test_format_chat_history_uses_recent_turns() -> None:
    history = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
        {"question": "Q3", "answer": "A3"},
    ]
    text = format_chat_history(history, max_turns=2)
    assert "Q1" not in text
    assert "Q2" in text
    assert "Q3" in text


def test_build_conversational_prompt_contains_sections() -> None:
    prompt = build_conversational_prompt("ctx", "What is this?", "User: hi")
    assert "Conversation history" in prompt
    assert "Context" in prompt


def test_build_query_rewrite_prompt_contains_current_question() -> None:
    prompt = build_query_rewrite_prompt("follow-up?", "User: prior")
    assert "follow-up?" in prompt
    assert "Rewritten query" in prompt


def test_build_self_reflection_prompt_contains_required_fields() -> None:
    prompt = build_self_reflection_prompt("q", "a", "c")
    assert "CONFIDENCE" in prompt
    assert "NEEDS_REWRITE" in prompt
    assert "REWRITE_QUERY" in prompt


def test_parse_reflection() -> None:
    text = (
        "CONFIDENCE: 0.82\n"
        "NEEDS_REWRITE: yes\n"
        "REWRITE_QUERY: standalone query\n"
        "RATIONALE: context is partial"
    )
    confidence, needs_rewrite, rewrite_query, rationale = parse_reflection(text)
    assert confidence == 0.82
    assert needs_rewrite is True
    assert rewrite_query == "standalone query"
    assert rationale == "context is partial"
