# Real Documents Pack (Public PDF Sources)

Bộ này dùng **tài liệu thật** từ nguồn công khai/chính thống và đã tối ưu cho app hiện tại (upload `pdf/docx`).

## Files

- `sources_real_docs.md`: danh sách nguồn PDF thật
- `download_real_docs.ps1`: script tải tài liệu thật về local
- `real_questions.md`: kịch bản test full capability
- `downloads/`: thư mục PDF dùng trực tiếp để upload

## Current cleanup status

- Đã loại bỏ bộ synthetic cũ (`data/sample_docs`, `data/realistic_docs`).
- Đã loại bỏ file `html/txt` trong bộ real docs vì app hiện chỉ nhận `pdf/docx`.

## Quick start (PowerShell)

```powershell
Set-Location "c:\Users\Admin\Desktop\ossd\data\real_docs"
./download_real_docs.ps1
```

## Test full capability (recommended order)

1. Chạy app và upload toàn bộ file trong `downloads/`.
2. Test nhóm A (ingestion + factual).
3. Test nhóm B (cross-document reasoning).
4. Test nhóm C (so sánh vector/keyword/hybrid).
5. Test nhóm D (metadata filter).
6. Test nhóm E (conversation memory follow-up).
7. Test nhóm F (out-of-scope safety).

## Notes

- Dùng `real_questions.md` làm checklist chấm pass/fail khi demo.
- Ưu tiên trích link nguồn gốc trong báo cáo để chứng minh dữ liệu thật.
