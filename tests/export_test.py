import io
import zipfile

import pytest

from export import _latin1_safe, letter_to_docx, letter_to_pdf

LETTER = "Dear team,\n\nI built “smart” systems — café résumé 日本語.\n\nThank you."


def test_latin1_safe_maps_and_never_leaves_wide_chars():
    out = _latin1_safe(LETTER + "\u0000\u0007 " + "x" * 200)
    assert all(ord(c) < 256 for c in out)
    assert '"smart"' in out and " - " in out and "\x00" not in out
    # unbreakable tokens get break points so the PDF engine can wrap them
    assert max(len(tok) for tok in out.split()) <= 60


def test_pdf_is_a_real_pdf_and_handles_hostile_text():
    pytest.importorskip("fpdf")
    data = letter_to_pdf(LETTER, "Ac‮me <b>Corp</b>")
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF") and len(data) > 500
    assert letter_to_pdf("", "X").startswith(b"%PDF-")                           # empty letter
    assert letter_to_pdf("w" * 5000, "X").startswith(b"%PDF-")                   # one giant token
    assert letter_to_pdf("a\n\n\n\nb\n\n" * 200, "X").startswith(b"%PDF-")        # many pages / blank runs


def test_docx_contains_heading_and_paragraphs():
    pytest.importorskip("docx")
    data = letter_to_docx("First paragraph.\n\nSecond paragraph.", "Acme")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    assert "Cover Letter" in xml and "Acme" in xml
    assert "First paragraph." in xml and "Second paragraph." in xml


def test_missing_optional_libraries_raise_import_error_not_crash(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "fpdf", None)
    monkeypatch.setitem(sys.modules, "docx", None)
    with pytest.raises(ImportError):
        letter_to_pdf("x", "y")
    with pytest.raises(ImportError):
        letter_to_docx("x", "y")
