from src.ui import _highlight_query_terms


def test_highlight_query_terms_marks_matching_word() -> None:
    text = "He thong ho tro truy xuat thong tin chinh xac"
    query = "truy xuat"

    highlighted = _highlight_query_terms(text, query)

    assert "<mark>truy</mark>" in highlighted.lower()
    assert "<mark>xuat</mark>" in highlighted.lower()


def test_highlight_query_terms_escapes_html() -> None:
    text = "<script>alert('x')</script> noi dung tai lieu"
    query = "noi dung"

    highlighted = _highlight_query_terms(text, query)

    assert "<script>" not in highlighted
    assert "&lt;script&gt;" in highlighted


def test_highlight_query_terms_can_also_use_answer_terms() -> None:
    text = "Tai lieu neu ro thoi gian khoi phuc dich vu la 4 gio."
    query = "khoi phuc"
    answer = "Thoi gian khoi phuc la 4 gio."

    highlighted = _highlight_query_terms(text, query, answer)

    assert "<mark>4</mark>" in highlighted
    assert "<mark>gio</mark>" in highlighted.lower()
