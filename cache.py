"""Small thread-safe TTL/LRU cache with optional disk persistence.

Used for query embeddings (memory only) and company research (memory + disk, 24h).
Keys are sha256 digests, so no raw input is used as a filename; values are plain JSON.
Callers must never put secrets in values (nothing here sees API keys).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from security import redact_secrets

_MISSING = object()


def app_dir() -> Path:
    """Resolved at call time so tests / users can redirect storage via JOBAGENT_HOME."""
    return Path(os.environ.get("JOBAGENT_HOME", Path.home() / ".job_application_agent"))


def make_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class TTLCache:
    def __init__(self, ttl_s: float = 3600.0, max_items: int = 512,
                 persist_path: Path | None = None, clock: Callable[[], float] = time.time):
        self.ttl_s = ttl_s
        self.max_items = max_items
        self.persist_path = Path(persist_path) if persist_path else None
        self._clock = clock
        self._lock = threading.RLock()
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._load()

    # -- persistence -----------------------------------------------------------
    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            now = self._clock()
            for k, (exp, val) in raw.items():
                if exp > now:
                    self._data[k] = (float(exp), val)
        except (OSError, ValueError, TypeError):
            self._data.clear()  # corrupt cache is just a cold cache

    def _flush(self) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.persist_path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({k: [exp, val] for k, (exp, val) in self._data.items()}, f)
            os.replace(tmp, self.persist_path)
        except (OSError, TypeError, ValueError):
            pass  # persistence is best-effort

    # -- api ---------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] <= self._clock():
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)
            self._flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._flush()


# --- embeddings (memory only) ---------------------------------------------------
_embed_cache = TTLCache(ttl_s=3600.0, max_items=256)


def cached_embed_query(embeddings: Any, model_key: str, text: str) -> Any:
    """embed_query with a per-process cache keyed on (model, text) — never on credentials."""
    key = make_key("embed", model_key, text)
    hit = _embed_cache.get(key, _MISSING)
    if hit is not _MISSING:
        return hit
    vec = list(embeddings.embed_query(text))
    _embed_cache.set(key, vec)
    return vec


# --- company research (memory + disk, 24h) ---------------------------------------
RESEARCH_TTL_S = 24 * 3600.0
_research_cache: TTLCache | None = None
_research_cache_dir: Path | None = None


def normalize_company(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def research_cache() -> TTLCache:
    global _research_cache, _research_cache_dir
    d = app_dir()
    if _research_cache is None or _research_cache_dir != d:
        _research_cache = TTLCache(ttl_s=RESEARCH_TTL_S, max_items=256,
                                   persist_path=d / "research_cache.json")
        _research_cache_dir = d
    return _research_cache


def get_cached_research(company: str) -> dict | None:
    if not normalize_company(company):
        return None
    return research_cache().get(make_key("research", normalize_company(company)))


def put_cached_research(company: str, value: dict) -> None:
    if not normalize_company(company):
        return
    # defence in depth: scrub anything key-shaped before it touches disk
    clean = {k: (redact_secrets(v) if isinstance(v, str) else v) for k, v in value.items()}
    research_cache().set(make_key("research", normalize_company(company)), clean)
