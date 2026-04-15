from src.prompts import build_prompt, is_vietnamese


def test_detect_vietnamese_true() -> None:
    assert is_vietnamese("Dự án này hỗ trợ tiếng Việt")


def test_detect_vietnamese_false() -> None:
    assert not is_vietnamese("This is an English sentence")


def test_build_prompt_vietnamese() -> None:
    prompt = build_prompt("Ngữ cảnh mẫu", "Mục tiêu dự án là gì?")
    assert "Trả lời" in prompt
    assert "Ngữ cảnh" in prompt


def test_build_prompt_english() -> None:
    prompt = build_prompt("Sample context", "What is the project goal?")
    assert "Answer" in prompt
    assert "Context" in prompt
