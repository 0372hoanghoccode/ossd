from __future__ import annotations

import re


_VIETNAMESE_CHAR_PATTERN = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def is_vietnamese(text: str) -> bool:
    return bool(_VIETNAMESE_CHAR_PATTERN.search(text))


def build_prompt(context: str, question: str) -> str:
    if is_vietnamese(question):
        return f"""Sử dụng ngữ cảnh sau để trả lời câu hỏi.
Nếu bạn không biết, chỉ cần nói rằng bạn không biết.
Trả lời ngắn gọn (3-4 câu) bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:"""

    return f"""Use the following context to answer the question.
If you don't know the answer, just say you don't know.
Keep the answer concise (3-4 sentences).

Context:
{context}

Question:
{question}

Answer:"""


def format_chat_history(chat_history: list[dict[str, str]], max_turns: int = 4) -> str:
    if not chat_history:
        return ""

    recent = chat_history[-max_turns:]
    lines: list[str] = []
    for turn in recent:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {turn.get('answer', '')}")
    return "\n".join(lines)


def build_conversational_prompt(context: str, question: str, chat_history: str) -> str:
    if is_vietnamese(question):
        return f"""Bạn là trợ lý hỏi đáp tài liệu.
Sử dụng lịch sử hội thoại và ngữ cảnh truy xuất để trả lời.
Nếu không đủ thông tin, nói rõ là bạn không biết.
Trả lời ngắn gọn (3-5 câu) bằng tiếng Việt.

Lịch sử hội thoại:
{chat_history or 'Không có'}

Ngữ cảnh:
{context}

Câu hỏi hiện tại:
{question}

Trả lời:"""

    return f"""You are a document QA assistant.
Use conversation history and retrieved context to answer.
If context is insufficient, say you don't know.
Keep response concise (3-5 sentences).

Conversation history:
{chat_history or 'None'}

Context:
{context}

Current question:
{question}

Answer:"""


def build_query_rewrite_prompt(question: str, chat_history: str) -> str:
    return f"""Rewrite the current user question into a standalone search query.
Preserve original intent, entities, and language.
Return only the rewritten query text.

Conversation history:
{chat_history or 'None'}

Current question:
{question}

Rewritten query:"""


def build_self_reflection_prompt(question: str, answer: str, context: str) -> str:
    return f"""Evaluate the answer quality based only on provided context.
Return exactly this format:
CONFIDENCE: <number between 0 and 1>
NEEDS_REWRITE: <yes/no>
REWRITE_QUERY: <single improved query or empty>
RATIONALE: <one short sentence>

Question: {question}

Answer: {answer}

Context:
{context}
"""


def parse_reflection(text: str) -> tuple[float, bool, str, str]:
    confidence_match = re.search(r"CONFIDENCE\s*:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
    confidence = min(1.0, max(0.0, confidence))

    needs_rewrite_match = re.search(r"NEEDS_REWRITE\s*:\s*(yes|no)", text, re.IGNORECASE)
    needs_rewrite = (needs_rewrite_match.group(1).lower() == "yes") if needs_rewrite_match else False

    rewrite_match = re.search(r"REWRITE_QUERY\s*:\s*(.*)", text, re.IGNORECASE)
    rewrite_query = rewrite_match.group(1).strip() if rewrite_match else ""

    rationale_match = re.search(r"RATIONALE\s*:\s*(.*)", text, re.IGNORECASE)
    rationale = rationale_match.group(1).strip() if rationale_match else ""

    return confidence, needs_rewrite, rewrite_query, rationale
