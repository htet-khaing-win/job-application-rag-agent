"""Letter export helpers (.docx and .pdf). Optional libraries are imported lazily."""
from __future__ import annotations

import io
import re
import unicodedata

from security import sanitize_text

# fpdf2's built-in fonts are latin-1 only; map common typographic characters first.
_PUNCT = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "…": "...", "•": "*", " ": " ", "−": "-", "​": "",
}
_LONG_TOKEN = re.compile(r"\S{60,}")


def letter_to_docx(letter: str, company: str) -> bytes:
    from docx import Document  # lazy: python-docx only needed for export
    doc = Document()
    doc.add_heading(f"Cover Letter — {sanitize_text(company, 120)}", level=1)
    for para in [p.strip() for p in letter.split("\n\n") if p.strip()]:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _latin1_safe(text: str) -> str:
    """Make text renderable with a core PDF font: map punctuation, decompose accents, replace the rest."""
    for src, dst in _PUNCT.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch if ord(ch) < 256 else (unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode() or "?")
                   for ch in text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # a single unbreakable token wider than the page makes fpdf2 raise; give it break points
    return _LONG_TOKEN.sub(lambda m: " ".join(m.group(0)[i:i + 40] for i in range(0, len(m.group(0)), 40)), text)


def letter_to_pdf(letter: str, company: str) -> bytes:
    """Render the letter as a simple A4 PDF. Raises ImportError if fpdf2 is not installed."""
    from fpdf import FPDF  # lazy: fpdf2 only needed for export

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, _latin1_safe(f"Cover Letter - {sanitize_text(company, 120)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    paragraphs = [p.strip() for p in (letter or "").split("\n\n") if p.strip()] or [""]
    for para in paragraphs:
        pdf.multi_cell(0, 6, _latin1_safe(para), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    return bytes(pdf.output())
