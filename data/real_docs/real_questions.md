# Real-Docs Question Set (Full Capability)

Áp dụng cho 3 file PDF thật trong `data/real_docs/downloads`:

- nist_sp_800_53r5.pdf
- nist_sp_800_61r2.pdf
- nist_sp_800_40r4.pdf

## A) Ingestion + Basic Factual

1. "What is the main purpose of NIST SP 800-53 Rev.5?"
2. "Summarize key phases from NIST SP 800-61 incident handling."
3. "What is enterprise patch management planning according to NIST SP 800-40 Rev.4?"

Expected:

- Trả lời bám đúng tài liệu nguồn.
- Có citation/source rõ ràng.

## B) Cross-document Reasoning

4. "How do incident response practices in 800-61 complement security controls in 800-53?"
5. "How does patch management guidance in 800-40 support risk reduction goals in 800-53?"

Expected:

- Có tổng hợp thông tin từ >=2 tài liệu.
- Không bịa nội dung không có trong nguồn.

## C) Retrieval Mode Comparison

Dùng cùng 1 câu hỏi: "prioritization of remediation actions based on risk"

6. Chạy ở mode `vector`
7. Chạy ở mode `keyword`
8. Chạy ở mode `hybrid`

Expected:

- `hybrid` thường cho grounding ổn định hơn.
- Citation chunk liên quan hơn so với chỉ keyword/vector.

## D) Metadata Filter Validation

9. Bật filter chỉ `nist_sp_800_61r2.pdf`, hỏi: "What are the incident response phases?"
10. Đổi filter sang `nist_sp_800_40r4.pdf`, hỏi lại cùng câu.

Expected:

- Câu trả lời đổi theo tài liệu đã lọc.
- Không trích dẫn chéo sai nguồn.

## E) Conversation Memory + Follow-up

11. Hỏi: "List 3 important control families from 800-53."
12. Hỏi tiếp: "Explain the second one in simpler terms and give one practical example."

Expected:

- Câu 12 hiểu được ngữ cảnh từ câu 11.
- Citation vẫn xuất hiện và đúng nguồn.

## F) Self-check / Safety

13. Hỏi ngoài phạm vi: "Write a full commercial contract for my startup."

Expected:

- Hệ thống từ chối/giới hạn hợp lý hoặc nói không đủ ngữ cảnh tài liệu.
- Không trả lời như thể có nguồn khi thực tế không có.
