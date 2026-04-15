# SmartDoc AI - Intelligent Document Q&A System

SmartDoc AI là hệ thống RAG chạy local cho phép tải lên PDF/DOCX và hỏi đáp bằng tiếng Việt/tiếng Anh.

## Tech Stack

- Streamlit (UI)
- LangChain + LangChain Community
- LangChain Ollama (`langchain-ollama`)
- FAISS (vector store)
- BM25 (keyword search) + Hybrid retrieval
- HuggingFace Embeddings: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Cross-Encoder re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Ollama + Qwen2.5

## Project Structure

```text
ossd/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ data/
├─ src/
│  ├─ config.py
│  ├─ pipeline.py
│  ├─ prompts.py
│  ├─ ui.py
│  └─ utils/logging.py
├─ tests/
│  └─ test_smoke.py
└─ documentation/
   ├─ project_report.tex
   └─ demo_checklist.md
```

## Setup (Windows)

1. Tạo virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Cài dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Cài Ollama và tải model:
   ```powershell
   ollama pull qwen2.5:7b
   ```
4. Chạy ứng dụng:
   ```powershell
   streamlit run app.py
   ```

## Usage

1. Upload file PDF hoặc DOCX.
2. Bấm **Process Document**.
3. Nhập câu hỏi và bấm **Get Answer**.
4. Xem câu trả lời và phần nguồn tham chiếu (top chunks).
5. Theo dõi lịch sử hội thoại ở sidebar.
6. Dùng **Clear History** hoặc **Clear Vector Store** (có xác nhận) khi cần reset.
7. Chọn `chunk_size` và `chunk_overlap` trong sidebar trước khi process để thử chiến lược chunk khác nhau.
8. Bấm **Run Chunk Benchmark** để so sánh nhanh các cấu hình chuẩn (500/1000/1500/2000 × 50/100/200).
9. Kiểm tra mục **Citations** để xem nguồn (file, page, chunk, start index), chọn xem context gốc và highlight theo từ khóa câu hỏi.
10. Chọn `retrieval_mode` (vector/hybrid/keyword) và bật `Conversational RAG`, `Re-ranking`, `Self-RAG reflection` trong sidebar.
11. Dùng bộ lọc metadata `source_name` và `doc_type` để giới hạn tài liệu truy xuất khi hỏi đáp multi-document.

## Configuration

Sao chép `.env.example` thành `.env` và chỉnh các biến nếu cần:

- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `RETRIEVER_K`, `RETRIEVER_SEARCH_TYPE`
- `HYBRID_ALPHA`, `BM25_FETCH_K`, `RERANK_TOP_N`, `CONVERSATIONAL_MEMORY_TURNS`
- `RERANK_MODEL`

## Testing

```powershell
pytest -q
```

## Notes

- Hệ thống chạy local 100%.
- Nếu lỗi kết nối model, kiểm tra Ollama service đang chạy.
- DOCX được trích xuất qua `python-docx`.
- Re-ranking dùng Cross-Encoder local; nếu model không tải được, hệ thống tự fallback sang ranking retrieval ban đầu.
