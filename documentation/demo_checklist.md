# Demo Checklist

## 1) Environment

- [ ] Python venv đã activate
- [ ] `pip install -r requirements.txt` thành công
- [ ] `ollama pull qwen2.5:7b` đã tải model
- [ ] Ollama đang chạy

## 2) Functional Demo

- [ ] Upload PDF hợp lệ
- [ ] Upload DOCX hợp lệ
- [ ] Process document thành công
- [ ] Hỏi câu tiếng Việt và nhận câu trả lời tiếng Việt
- [ ] Hỏi câu tiếng Anh và nhận câu trả lời tiếng Anh
- [ ] Câu hỏi ngoài ngữ cảnh trả lời “không biết”/“I don't know”
- [ ] Hiển thị nguồn tham chiếu (top chunks)
- [ ] Sidebar hiển thị lịch sử hội thoại
- [ ] Nút Clear History mở hộp xác nhận và xóa thành công
- [ ] Nút Clear Vector Store mở hộp xác nhận và xóa index thành công
- [ ] Hiển thị citations với metadata (file, page, chunk, start index)
- [ ] Chọn được từng nguồn để xem context gốc
- [ ] Đoạn context có highlight theo từ khóa câu hỏi
- [ ] Upload nhiều tài liệu cùng lúc và trả lời được từ nhiều nguồn
- [ ] Lọc metadata theo source_name/doc_type hoạt động đúng
- [ ] Chuyển retrieval_mode giữa vector/hybrid/keyword và thấy khác biệt kết quả
- [ ] Conversational RAG xử lý đúng câu hỏi follow-up
- [ ] Bật re-ranking và xác nhận thứ tự chunks thay đổi hợp lý
- [ ] Self-RAG trả về confidence/rationale và có thể rewrite query khi cần

## 3) Error Handling

- [ ] Không upload file mà bấm process
- [ ] Nhập câu hỏi khi chưa process tài liệu
- [ ] Tắt Ollama và xác nhận app báo lỗi rõ ràng

## 4) Performance Notes

- [ ] Ghi lại thời gian process tài liệu
- [ ] Ghi lại thời gian retrieve và generate cho ít nhất 3 câu hỏi
- [ ] So sánh kết quả benchmark giữa các cấu hình chunk_size/chunk_overlap
- [ ] Chọn cấu hình chunk tối ưu cho demo và ghi lại lý do
