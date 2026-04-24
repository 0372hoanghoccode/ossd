from __future__ import annotations

import json
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


def build_corag_decompose_prompt(question: str) -> str:
    if is_vietnamese(question):
        return f"""Tách câu hỏi thành các phần thông tin nguyên tử cần truy xuất cho CoRAG.
Yêu cầu:
- Tối đa 5 phần.
- Mỗi phần phải có thể truy xuất độc lập bằng tìm kiếm.
- KHÔNG tách rời thực thể tên riêng (ví dụ tên đội, tên model, tên người, tên cluster).
- Mỗi phần nên là cụm từ truy vấn, không phải câu dài.
- Trả về DUY NHẤT JSON hợp lệ theo đúng schema: {{"parts": ["part1", "part2"]}}
- Không thêm markdown, không thêm giải thích.

Câu hỏi:
{question}
"""

    return f"""Decompose the question into atomic, independently retrievable parts for CoRAG.
Requirements:
- Maximum 5 parts.
- Each part must be independently searchable.
- DO NOT split proper nouns or named entities (team names, model names, person names, cluster names).
- Each part should be a search-worthy phrase, not a full sentence.
- Return ONLY valid JSON with this exact schema: {{"parts": ["part1", "part2"]}}
- No markdown, no explanation.

Question:
{question}
"""


def build_corag_evaluator_prompt(
    question: str,
    required_parts: list[str],
    context: str,
    discovered_entities: dict[str, str],
) -> str:
    required_parts_json = json.dumps(required_parts, ensure_ascii=False)
    discovered_json = json.dumps(discovered_entities, ensure_ascii=False)

    if is_vietnamese(question):
        return f"""Bạn là evaluator cho TRUE CoRAG.
Nhiệm vụ chính: đánh giá bằng chứng theo từng required part và tạo next_query có neo thực thể cụ thể.

Đầu vào:
- question: {question}
- required_parts: {required_parts_json}
- discovered_entities (đã tìm được từ các bước trước): {discovered_json}

Quy tắc bắt buộc:
1) Với MỖI required part, chỉ đánh dấu covered=true nếu context có bằng chứng TƯỜNG MINH
   (tên cụ thể, số cụ thể, ngày cụ thể, giá trị cụ thể).
2) sufficient=true CHỈ khi tất cả required parts đều covered=true.
3) Không được dùng kiến thức nền của mô hình để lấp chỗ trống.
4) Khi còn thiếu: chọn missing part quan trọng nhất và tạo next_query DUY NHẤT có neo vào thực thể cụ thể đã biết.
   - Truy vấn mơ hồ là sai.
   - Truy vấn đúng phải chứa entity cụ thể từ discovered_entities hoặc context hiện tại.
5) Trích xuất new_entities phát hiện được ở bước này dưới dạng dict key-value.

Trả về DUY NHẤT JSON hợp lệ, không markdown, không giải thích:
{{
  "sufficient": false,
  "evidence_map": [
    {{"part": "...", "covered": true, "value": "..."}},
    {{"part": "...", "covered": false, "value": null}}
  ],
  "missing_parts": ["..."],
  "next_query": "...",
  "new_entities": {{"...": "..."}},
  "reasoning": "..."
}}

Context:
{context}
"""

    return f"""You are the evaluator for TRUE CoRAG.
Primary task: verify explicit evidence per required part and produce an entity-grounded next_query.

Inputs:
- question: {question}
- required_parts: {required_parts_json}
- discovered_entities from previous steps: {discovered_json}

Hard rules:
1) For EACH required part, set covered=true only when explicit evidence exists in context
   (specific name, number, date, or value).
2) sufficient=true ONLY when all required parts are explicitly covered.
3) Never use background knowledge to fill missing facts.
4) If something is missing: pick the most important missing part and produce ONE next_query
   grounded on concrete known entities from discovered_entities or current context.
   - Vague pronoun-based queries are invalid.
5) Extract newly discovered entities in this step as key-value pairs in new_entities.

Return ONLY valid JSON (no markdown, no explanations):
{{
  "sufficient": false,
  "evidence_map": [
    {{"part": "...", "covered": true, "value": "..."}},
    {{"part": "...", "covered": false, "value": null}}
  ],
  "missing_parts": ["..."],
  "next_query": "...",
  "new_entities": {{"...": "..."}},
  "reasoning": "..."
}}

Context:
{context}
"""


def build_corag_final_answer_prompt(
    question: str,
    context: str,
    language: str = "vi",
) -> str:
    lang = language
    if lang not in {"vi", "en"}:
        lang = "vi" if is_vietnamese(question) else "en"

    if lang == "vi":
        return f"""Trả lời câu hỏi chỉ dựa trên context bên dưới.
Yêu cầu:
- Trả lời ngắn gọn, đúng sự kiện.
- Trình bày theo từng ý thông tin cần thiết.
- Nếu thiếu dữ liệu cho phần nào, ghi rõ: "Không tìm thấy: <phần thiếu>".
- Không dùng kiến thức ngoài context.

Câu hỏi:
{question}

Context:
{context}

Trả lời:
"""

    return f"""Answer the question using ONLY the context below.
Requirements:
- Keep the response concise and factual.
- Structure answer by required facts.
- If any part is missing, explicitly state: "Not found: <missing part>".
- Do not use external/background knowledge.

Question:
{question}

Context:
{context}

Answer:
"""
