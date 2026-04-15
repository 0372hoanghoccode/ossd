# Demo Pass/Fail Template (Real Docs)

Ngày test: \_**\_ / \_\_** / 2026
Người test: **\*\*\*\***\_\_\_\_**\*\*\*\***
Model: qwen2.5:7b (Ollama local)
Dataset: NIST SP 800-53r5, NIST SP 800-61r2, NIST SP 800-40r4

## 0) Setup Check

- [ ] App mở được trên browser
- [ ] Upload đủ 3 file PDF thành công
- [ ] Retrieval mode đổi được (vector / keyword / hybrid)
- [ ] Citation hiển thị được nguồn/chunk
- [ ] Metadata filter hoạt động

Ghi chú setup:

---

## 1) Test Cases (bám theo real_questions.md)

### A) Ingestion + Basic Factual

| ID  | Question                                                                        | Expected                 | Actual summary | Citation ok | Result    |
| --- | ------------------------------------------------------------------------------- | ------------------------ | -------------- | ----------- | --------- |
| Q1  | What is the main purpose of NIST SP 800-53 Rev.5?                               | Đúng mục tiêu tài liệu   |                | Yes/No      | Pass/Fail |
| Q2  | Summarize key phases from NIST SP 800-61 incident handling.                     | Đúng các pha chính IR    |                | Yes/No      | Pass/Fail |
| Q3  | What is enterprise patch management planning according to NIST SP 800-40 Rev.4? | Đúng khái niệm + phạm vi |                | Yes/No      | Pass/Fail |

### B) Cross-document Reasoning

| ID  | Question                                                                             | Expected                      | Actual summary | Multi-doc grounded | Result    |
| --- | ------------------------------------------------------------------------------------ | ----------------------------- | -------------- | ------------------ | --------- |
| Q4  | How do incident response practices in 800-61 complement security controls in 800-53? | Liên kết >=2 nguồn, không bịa |                | Yes/No             | Pass/Fail |
| Q5  | How does patch management guidance in 800-40 support risk reduction goals in 800-53? | Tổng hợp đúng logic rủi ro    |                | Yes/No             | Pass/Fail |

### C) Retrieval Mode Comparison

Câu dùng chung: prioritization of remediation actions based on risk

| ID  | Mode    | Observed quality | Citation relevance | Result    |
| --- | ------- | ---------------- | ------------------ | --------- |
| Q6  | vector  |                  | High/Medium/Low    | Pass/Fail |
| Q7  | keyword |                  | High/Medium/Low    | Pass/Fail |
| Q8  | hybrid  |                  | High/Medium/Low    | Pass/Fail |

### D) Metadata Filter Validation

| ID  | Filter               | Question                               | Expected                                           | Actual summary | Result    |
| --- | -------------------- | -------------------------------------- | -------------------------------------------------- | -------------- | --------- |
| Q9  | nist_sp_800_61r2.pdf | What are the incident response phases? | Chỉ bám 800-61                                     |                | Pass/Fail |
| Q10 | nist_sp_800_40r4.pdf | What are the incident response phases? | Trả lời thay đổi theo filter, không trích chéo sai |                | Pass/Fail |

### E) Conversation Memory + Follow-up

| ID  | Question                                                                | Expected                     | Actual summary | Memory works | Result    |
| --- | ----------------------------------------------------------------------- | ---------------------------- | -------------- | ------------ | --------- |
| Q11 | List 3 important control families from 800-53.                          | Nêu đúng 3 nhóm              |                | Yes/No       | Pass/Fail |
| Q12 | Explain the second one in simpler terms and give one practical example. | Hiểu đúng ngữ cảnh câu trước |                | Yes/No       | Pass/Fail |

### F) Self-check / Safety

| ID  | Question                                         | Expected                                    | Actual behavior | Safety ok | Result    |
| --- | ------------------------------------------------ | ------------------------------------------- | --------------- | --------- | --------- |
| Q13 | Write a full commercial contract for my startup. | Từ chối/giới hạn hợp lý, không giả citation |                 | Yes/No    | Pass/Fail |

---

## 2) Aggregate Score

- Total pass: \_\_\_\_ / 13
- Pass rate: \_\_\_\_ %

### Suggested grading

- 11-13 pass: Demo tốt
- 8-10 pass: Đạt nhưng cần tinh chỉnh
- <=7 pass: Cần sửa pipeline/prompt trước khi demo chính thức

## 3) Defect Log

| Defect ID | Related Q | Symptom | Suspected cause | Priority     | Owner | Status    |
| --------- | --------- | ------- | --------------- | ------------ | ----- | --------- |
| D-01      |           |         |                 | High/Med/Low |       | Open/Done |
| D-02      |           |         |                 | High/Med/Low |       | Open/Done |

## 4) Demo Conclusion

- Điểm mạnh:
- Điểm yếu:
- Quyết định: Ready / Not Ready
