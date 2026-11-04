"""Local history of cover letters the user was satisfied with.

A small SQLite file (``history.db`` next to the vault) holds the job description, the letter and a few
non-sensitive metrics per entry. Entries are listed newest first, ten per page, and can be deleted
PERMANENTLY.

Design notes
* Journal mode is ``DELETE`` (not WAL). In WAL mode the old page images of a deleted letter live on in
  ``history.db-wal`` until a checkpoint, plus a shared-memory file; with the rollback journal there is no
  long-lived sibling file at all (the journal is removed at commit). For a tiny, rarely-written database the
  extra WAL concurrency is not worth a second place where deleted text can hide. Concurrency across several
  ``HistoryStore`` instances (Streamlit re-runs create new ones) comes from ``busy_timeout`` plus
  ``BEGIN IMMEDIATE`` write transactions.
* ``PRAGMA secure_delete=ON`` makes SQLite overwrite deleted content with zeros, and a ``VACUUM`` after every
  delete rebuilds the file so freed pages cannot keep stale text. If the VACUUM cannot get its exclusive lock
  (another instance is reading), the zeroing has already happened and the compaction is simply retried on
  the next delete. Limit: this is file-level erasure. It cannot defeat forensic recovery of blocks the
  filesystem has already released (the unlinked rollback journal, SSD wear levelling, backups).
* Inputs are sanitised and API-key-shaped strings are redacted BEFORE anything is written, so a key pasted
  into a job description can never reach the database.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from security import MAX_JD_CHARS, redact_secrets, sanitize_text

PAGE_SIZE = 10
MAX_LIMIT = PAGE_SIZE * 5
MAX_COMPANY = 120
MAX_TITLE = 200
MAX_LETTER = 20_000
MAX_META_BYTES = 2048
MAX_RUN_ID = 32
BUSY_TIMEOUT_MS = 5000
SCRUB_ATTEMPTS = 5

SCHEMA_VERSION = 1


class HistoryError(RuntimeError):
    """The history database is corrupt, locked, closed or otherwise unusable."""


@dataclass(frozen=True)
class Entry:
    id: int
    created_at: str        # ISO-8601 UTC, e.g. "2026-11-04T03:27:29Z"
    company: str
    job_title: str
    jd: str
    letter: str
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------------------------- schema
def _migrate_to_1(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS letters ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "created_at TEXT NOT NULL, "
        "company TEXT NOT NULL, "
        "job_title TEXT NOT NULL, "
        "jd TEXT NOT NULL, "
        "letter TEXT NOT NULL, "
        "meta TEXT NOT NULL DEFAULT '{}')"
    )
    # id is part of the index so that entries created within the same second still sort newest-first.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_letters_created ON letters(created_at DESC, id DESC)")


# version -> function that upgrades a database from version-1 to that version (applied in order)
_MIGRATIONS: Dict[int, Callable[[sqlite3.Connection], None]] = {1: _migrate_to_1}


# ---------------------------------------------------------------------------------------------- helpers
def _default_path() -> Path:
    """<JOBAGENT_HOME or ~/.job_application_agent>/history.db, resolved when called (not at import)."""
    home = os.environ.get("JOBAGENT_HOME")
    base = Path(home) if home else Path.home() / ".job_application_agent"
    return base / "history.db"


def _create_private_file(path: Path) -> None:
    """Create an empty file readable only by the owner. Best-effort: never fails on Windows."""
    if path.exists():
        return
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    except OSError:
        return
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass


def _clean(text: Any, name: str, max_chars: int, single_line: bool = False) -> str:
    """Sanitise, redact secrets, THEN cap (capping first could leave half of a key behind)."""
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a string")
    cleaned = redact_secrets(sanitize_text(text, max_chars * 4))
    if single_line:
        cleaned = " ".join(cleaned.split())
    return cleaned[:max_chars].strip()


def _clean_meta(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep only the known non-sensitive keys with the right types; silently drop everything else."""
    if meta is None:
        return {}
    if not isinstance(meta, Mapping):
        raise ValueError("meta must be a mapping")
    out: Dict[str, Any] = {}

    def num(v: Any) -> Optional[float]:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return None
        return float(v)

    score = num(meta.get("retrieval_score"))
    if score is not None:
        out["retrieval_score"] = score
    cost = num(meta.get("cost_usd"))
    if cost is not None:
        out["cost_usd"] = cost
    ref = meta.get("refinements")
    if isinstance(ref, int) and not isinstance(ref, bool):
        out["refinements"] = ref
    ver = meta.get("verified")
    if isinstance(ver, bool):
        out["verified"] = ver
    rid = meta.get("run_id")
    if isinstance(rid, str):
        rid = sanitize_text(rid, MAX_RUN_ID + 1)
        if 0 < len(rid) <= MAX_RUN_ID:
            out["run_id"] = rid
    if len(json.dumps(out).encode("utf-8")) > MAX_META_BYTES:   # cannot happen with the keys above; defensive
        return {}
    return out


def _row_to_entry(row: Any) -> Entry:
    try:
        meta = json.loads(row[6])
        if not isinstance(meta, dict):
            meta = {}
    except (TypeError, ValueError):
        meta = {}
    return Entry(id=int(row[0]), created_at=row[1], company=row[2], job_title=row[3], jd=row[4],
                 letter=row[5], meta=meta)


_SELECT = "SELECT id, created_at, company, job_title, jd, letter, meta FROM letters"


# ---------------------------------------------------------------------------------------------- store
class HistoryStore:
    """Thread-safe SQLite store. Create as many instances on the same file as you like."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else _default_path()
        self._lock = threading.RLock()
        self._now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)
        self._conn: Optional[sqlite3.Connection] = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _create_private_file(self.path)
            self._conn = self._connect()
            self._init_schema()
        except sqlite3.Error as e:
            self._abandon()
            raise HistoryError(f"Cannot open history database {self.path.name}: {e}") from e
        except OSError as e:
            self._abandon()
            raise HistoryError(f"Cannot create history database at {self.path}: {e}") from e
        except HistoryError:
            self._abandon()
            raise

    # -- plumbing ----------------------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: autocommit, so we control transactions explicitly (VACUUM cannot run inside one)
        conn = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None,
                               check_same_thread=False)
        conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_MS)}")
        conn.execute("PRAGMA secure_delete = ON")       # overwrite deleted content with zeros
        conn.execute("PRAGMA journal_mode = DELETE")    # no long-lived -wal / -shm files (see module docstring)
        return conn

    def _abandon(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise HistoryError("History store is closed.")
        return self._conn

    def _init_schema(self) -> None:
        conn = self._db()
        conn.execute("BEGIN IMMEDIATE")     # serialises two instances racing to create the schema
        try:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise HistoryError(f"History database is from a newer version (schema {current}); "
                                   f"this build understands up to {SCHEMA_VERSION}.")
            for version in range(current + 1, SCHEMA_VERSION + 1):
                _MIGRATIONS[version](conn)
                conn.execute(f"PRAGMA user_version = {int(version)}")
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    def _write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run `fn` inside BEGIN IMMEDIATE ... COMMIT; translate sqlite errors into HistoryError."""
        with self._lock:
            conn = self._db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(conn)
                    conn.execute("COMMIT")
                    return result
                except BaseException:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
            except sqlite3.Error as e:
                raise HistoryError(f"History database error: {e}") from e

    def _scrub(self) -> None:
        """Compact the file so deleted text cannot survive in freed pages. Retries while another instance reads."""
        conn = self._db()
        for attempt in range(SCRUB_ATTEMPTS):
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # no-op in DELETE mode; covers a file left in WAL
                conn.execute("VACUUM")
                return
            except sqlite3.OperationalError:
                if attempt == SCRUB_ATTEMPTS - 1:
                    return    # secure_delete already zeroed the content; the next delete retries the compaction
                time.sleep(0.1 * (attempt + 1))
            except sqlite3.Error as e:
                raise HistoryError(f"History database error: {e}") from e

    # -- public API --------------------------------------------------------------------------------
    def add(self, company: str, job_title: str, jd: str, letter: str,
            meta: Optional[Mapping[str, Any]] = None) -> int:
        company_c = _clean(company, "company", MAX_COMPANY, single_line=True)
        title_c = _clean(job_title, "job_title", MAX_TITLE, single_line=True)
        jd_c = _clean(jd, "jd", MAX_JD_CHARS)
        letter_c = _clean(letter, "letter", MAX_LETTER)
        if not company_c:
            raise ValueError("company must not be empty")
        if not letter_c:
            raise ValueError("letter must not be empty")
        meta_json = json.dumps(_clean_meta(meta), separators=(",", ":"))
        created = self._now().astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def insert(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "INSERT INTO letters (created_at, company, job_title, jd, letter, meta) VALUES (?, ?, ?, ?, ?, ?)",
                (created, company_c, title_c, jd_c, letter_c, meta_json))
            return int(cur.lastrowid or 0)

        return int(self._write(insert))

    def list(self, limit: int = PAGE_SIZE, offset: int = 0) -> List[Entry]:
        lim = max(1, min(int(limit), MAX_LIMIT))
        off = max(0, int(offset))
        with self._lock:
            try:
                rows = self._db().execute(
                    _SELECT + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?", (lim, off)).fetchall()
            except sqlite3.Error as e:
                raise HistoryError(f"History database error: {e}") from e
        return [_row_to_entry(r) for r in rows]

    def get(self, entry_id: int) -> Optional[Entry]:
        with self._lock:
            try:
                row = self._db().execute(_SELECT + " WHERE id = ?", (int(entry_id),)).fetchone()
            except sqlite3.Error as e:
                raise HistoryError(f"History database error: {e}") from e
        return _row_to_entry(row) if row is not None else None

    def count(self) -> int:
        with self._lock:
            try:
                return int(self._db().execute("SELECT COUNT(*) FROM letters").fetchone()[0])
            except sqlite3.Error as e:
                raise HistoryError(f"History database error: {e}") from e

    def delete(self, entry_id: int) -> bool:
        """PERMANENTLY remove one entry. Returns True if a row was removed."""
        eid = int(entry_id)
        removed = int(self._write(lambda c: c.execute("DELETE FROM letters WHERE id = ?", (eid,)).rowcount))
        if removed:
            with self._lock:
                self._scrub()
        return removed > 0

    def delete_all(self) -> int:
        """PERMANENTLY remove every entry. Returns how many were removed."""
        removed = int(self._write(lambda c: c.execute("DELETE FROM letters").rowcount))
        if removed:
            with self._lock:
                self._scrub()
        return removed

    def close(self) -> None:
        with self._lock:
            self._abandon()

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
