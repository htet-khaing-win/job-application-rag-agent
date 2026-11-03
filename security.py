"""Security primitives: secret redaction, input sanitising, upload validation."""
from __future__ import annotations

import io
import logging
import os
import re
import time
import unicodedata
import zipfile
from typing import Tuple

MAX_JD_CHARS = 20_000
MAX_COMPANY_CHARS = 120
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_UPLOAD_EXTS = {".pdf", ".docx", ".txt"}

# Patterns for well-known API key shapes. Used to scrub logs, errors and UI output.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),            # OpenAI / Anthropic style
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),           # Google
    re.compile(r"tvly-[A-Za-z0-9_\-]{12,}"),          # Tavily
    re.compile(r"pcsk_[A-Za-z0-9_\-]{16,}"),          # Pinecone
    re.compile(r"(?i)(?<=bearer )[A-Za-z0-9_\-\.=+/]{16,}"),   # opaque bearer tokens / JWTs
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token)(\"?\s*[:=]\s*\"?)([A-Za-z0-9_\-\.]{12,})"),
]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮⁠-⁤]")

_INJECTION_MARKERS = re.compile(
    r"(?i)(ignore (all |any )?(previous|prior|above) (instructions|prompts)|"
    r"disregard (the )?(system|previous) (prompt|instructions)|"
    r"you are now |reveal (your )?(system )?prompt|print (the )?api[_ ]?key)"
)


def redact_secrets(text: object) -> str:
    """Replace anything that looks like an API key with a placeholder."""
    out = str(text)
    for pat in _SECRET_PATTERNS[:-1]:
        out = pat.sub("[REDACTED]", out)
    out = _SECRET_PATTERNS[-1].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
    return out


def mask_key(key: str) -> str:
    """Show only the last 4 chars of a key for UI display."""
    if not key:
        return ""
    return "•" * 8 + key[-4:] if len(key) > 8 else "•" * len(key)


class SecretRedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_secrets(record.getMessage())
            record.args = ()
        except Exception:  # never let logging crash the app
            pass
        return True


def install_log_redaction() -> None:
    root = logging.getLogger()
    if not any(isinstance(f, SecretRedactingFilter) for f in root.filters):
        root.addFilter(SecretRedactingFilter())
    for h in root.handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in h.filters):
            h.addFilter(SecretRedactingFilter())


def sanitize_text(text: str, max_chars: int = MAX_JD_CHARS) -> str:
    """Normalise unicode, strip control / bidi characters and cap length."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _CONTROL_RE.sub(" ", text.replace("\r\n", "\n"))
    return text.strip()[:max_chars]


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_MARKERS.search(text or ""))


def fence_untrusted(label: str, text: str) -> str:
    """Wrap untrusted content so prompts can instruct the model to treat it as data only."""
    safe = text.replace("</", "<​/")  # prevent closing-tag spoofing
    return f"<{label}>\n{safe}\n</{label}>"


def safe_filename(name: str) -> str:
    """Strip any path components and unsafe chars from a user-supplied filename."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = unicodedata.normalize("NFKC", base)
    base = re.sub(r"[^A-Za-z0-9._\- ]", "_", base).strip(" .")
    if not base or base in {".", ".."}:
        raise ValueError("Invalid filename.")
    return base[:120]


_MAGIC = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),
}


MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024
MAX_DOCX_ENTRIES = 2000
MAX_DOCX_RATIO = 100


def _docx_problem(data: bytes) -> str:
    """Return a rejection reason for a malformed or decompression-bomb .docx, else ''."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            names = {i.filename for i in infos}
            if "word/document.xml" not in names:
                return "Not a valid .docx file."
            if len(infos) > MAX_DOCX_ENTRIES:
                return "Document has too many parts."
            if sum(i.file_size for i in infos) > MAX_DOCX_UNCOMPRESSED:
                return "Document expands too large."
            for i in infos:
                if i.filename.startswith(("/", "..")) or ".." in i.filename.split("/"):
                    return "Document contains unsafe paths."
                if i.file_size > 1024 * 1024 and i.file_size / max(1, i.compress_size) > MAX_DOCX_RATIO:
                    return "Document compression ratio is suspicious."
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError):
        return "Not a valid .docx file."
    return ""


def replace_with_retry(src: str, dst: "str | os.PathLike[str]", attempts: int = 20) -> None:
    """os.replace that tolerates transient PermissionError (Windows file locking under concurrent writers)."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.01 * (i + 1))


def validate_upload(filename: str, data: bytes) -> Tuple[bool, str]:
    """Validate extension, size and magic bytes of an uploaded resume."""
    try:
        name = safe_filename(filename)
    except ValueError as e:
        return False, str(e)
    ext = os.path.splitext(name)[-1].lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        return False, "Unsupported file format (pdf, docx, txt only)."
    if not data:
        return False, "File is empty."
    if len(data) > MAX_UPLOAD_BYTES:
        return False, "File too large (max 5MB)."
    if ext in _MAGIC and not data.startswith(_MAGIC[ext]):
        return False, "File content does not match its extension."
    if ext == ".docx":
        problem = _docx_problem(data)
        if problem:
            return False, problem
    if ext == ".txt":
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return False, "Text files must be UTF-8."
    return True, "ok"
