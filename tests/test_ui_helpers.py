from src.ui import _highlight_query_terms


def test_highlight_query_terms_marks_matching_word() -> None:
    text = "Hệ thống hỗ trợ truy xuất thông tin chính xác"
    query = "truy xuất"

    highlighted = _highlight_query_terms(text, query)

    assert "<mark>truy</mark>" in highlighted.lower()
    assert "<mark>xuất</mark>" in highlighted.lower()


def test_highlight_query_terms_escapes_html() -> None:
    text = "<script>alert('x')</script> nội dung tài liệu"
    query = "nội dung"

    highlighted = _highlight_query_terms(text, query)

    assert "<script>" not in highlighted
    assert "&lt;script&gt;" in highlighted
