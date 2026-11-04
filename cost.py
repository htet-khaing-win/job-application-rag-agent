"""Cost estimation and per-key monthly spend ledger.

PRICES are ESTIMATES in USD per 1M tokens (input, output). Provider pricing changes —
review and update this table periodically; unknown/local models cost 0.0.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# model-name prefix -> (usd per 1M input tokens, usd per 1M output tokens). ESTIMATES: update me.
PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
}

# Documented heuristic: (fixed prompt tokens, fraction of JD tokens included, output tokens).
# Tokens ~= chars / 4. Primary binding of each chat role only.
ROLE_TOKEN_HEURISTIC: Dict[str, Tuple[int, float, int]] = {
    "jd_analyst": (350, 1.0, 400),
    "writer": (900, 0.6, 450),
    "critic": (700, 0.4, 250),
    "editor": (500, 0.0, 450),
    "verifier": (600, 0.0, 450),
}


from security import replace_with_retry  # noqa: E402


class BudgetExceeded(RuntimeError):
    """Raised when a key's monthly budget is exhausted."""


def _price_for(model: str) -> Optional[Tuple[float, float]]:
    best: Optional[str] = None
    for prefix in PRICES:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return PRICES[best] if best else None


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    """USD estimate; 0.0 for local (ollama-style tags) and unknown models."""
    price = _price_for(model or "")
    if price is None:
        return 0.0
    return round((max(in_tok, 0) * price[0] + max(out_tok, 0) * price[1]) / 1_000_000, 6)


def estimate_run_cost(profile: Any, jd_chars: int) -> Dict[str, Any]:
    """Pre-run estimate from the primary binding of every chat role."""
    jd_tokens = max(jd_chars, 0) // 4
    per_role: Dict[str, Dict[str, Any]] = {}
    total_usd = 0.0
    total_tokens = 0
    for role, (fixed, jd_frac, out_tok) in ROLE_TOKEN_HEURISTIC.items():
        bindings = profile.roles.get(role)
        if not bindings:
            continue
        b = bindings[0]
        in_tok = fixed + int(jd_tokens * jd_frac)
        usd = 0.0 if b.provider == "ollama" else estimate_cost(b.model, in_tok, out_tok)
        per_role[role] = {"model": f"{b.provider}:{b.model}", "in_tok": in_tok, "out_tok": out_tok, "usd": usd}
        total_usd += usd
        total_tokens += in_tok + out_tok
    return {"per_role": per_role, "total_usd": round(total_usd, 6), "total_tokens": total_tokens}


def _month_key(now: Optional[dt.datetime] = None) -> str:
    return (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m")


class SpendLedger:
    """usage.json: {"YYYY-MM": {alias: usd}}; thread-safe, atomic writes."""

    _locks: Dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, app_dir: Optional[Path] = None):
        if app_dir is None:
            from vault import APP_DIR
            app_dir = APP_DIR
        self.app_dir = Path(app_dir)
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.app_dir / "usage.json"
        with SpendLedger._locks_guard:  # one lock per file, shared across instances
            self._lock = SpendLedger._locks.setdefault(str(self._path.resolve()), threading.RLock())

    def _read(self) -> Dict[str, Dict[str, float]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: Dict[str, Dict[str, float]]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.app_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            replace_with_retry(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def add(self, alias: str, usd: float, now: Optional[dt.datetime] = None) -> None:
        if usd <= 0:
            return
        with self._lock:
            data = self._read()
            month = data.setdefault(_month_key(now), {})
            month[alias] = round(float(month.get(alias, 0.0)) + float(usd), 6)
            self._write(data)

    def month_total(self, alias: str, now: Optional[dt.datetime] = None) -> float:
        with self._lock:
            return float(self._read().get(_month_key(now), {}).get(alias, 0.0))

    def check(self, alias: str, cap: Optional[float], now: Optional[dt.datetime] = None) -> None:
        if cap is None:
            return
        spent = self.month_total(alias, now)
        if spent >= cap:
            raise BudgetExceeded(f"Monthly budget reached for key '{alias}' (${spent:.2f} of ${cap:.2f}).")
