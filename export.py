"""Letter export helpers."""
from __future__ import annotations

import io

from security import sanitize_text


def letter_to_docx(letter: str, company: str) -> bytes:
    from docx import Document  # lazy: python-docx only needed for export
    doc = Document()
    doc.add_heading(f"Cover Letter — {sanitize_text(company, 120)}", level=1)
    for para in [p.strip() for p in letter.split("\n\n") if p.strip()]:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
