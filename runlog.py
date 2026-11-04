"""Structured, privacy-preserving run log (JSONL) with percentile summaries.

Only an allow-listed set of fields is ever written: no letter, JD, resume or company
free-text, and no keys. Everything is passed through secret redaction as a last line of defence.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from cache import app_dir
from security import redact_secrets

MAX_BYTES = 5 * 1024 * 1024
_ALLOWED = {"run_id", "ts", "git_sha", "models", "node_timings", "usage", "cost_usd", "tokens_in",
            "tokens_out", "healing_events", "ablate", "ok", "fallback", "error_type", "duration_ms",
            "refinement_count", "letter_verified"}
_lock = threading.Lock()
_sha_cache: str | None = None


def _git_sha() -> str:
    global _sha_cache
    if _sha_cache is None:
        try:
            _sha_cache = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=2).strip()  # noqa: S607
        except Exception:  # noqa: BLE001
            _sha_cache = "unknown"
    return _sha_cache


def log_path() -> Path:
    return app_dir() / "runs.jsonl"


def _scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, dict):
        return {str(k): _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def _rotate(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size >= MAX_BYTES:
            backup = path.with_suffix(".jsonl.1")
            os.replace(path, backup)
    except OSError:
        pass


def log_run(entry: dict[str, Any], path: Path | None = None) -> None:
    path = Path(path) if path else log_path()
    safe = {k: v for k, v in entry.items() if k in _ALLOWED}
    safe.setdefault("ts", dt.datetime.now(dt.UTC).isoformat())
    safe.setdefault("git_sha", _git_sha())
    line = json.dumps(_scrub(safe), default=str, separators=(",", ":"))
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_runs(limit: int | None = None, path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else log_path()
    if not path.exists():
        return []
    runs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                runs.append(json.loads(line))
            except ValueError:
                continue  # skip a torn/corrupt line
    return runs[-limit:] if limit else runs


def percentiles(values: Sequence[float], pcts: Iterable[int] = (50, 95)) -> dict[int, float]:
    """Nearest-rank percentiles."""
    xs = sorted(values)
    if not xs:
        return {p: 0.0 for p in pcts}
    out = {}
    for p in pcts:
        rank = max(1, math.ceil(p / 100 * len(xs)))
        out[p] = xs[min(rank, len(xs)) - 1]
    return out


def summarize_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_node: dict[str, list[float]] = {}
    for r in runs:
        for t in r.get("node_timings", []) or []:
            per_node.setdefault(t["node"], []).append(float(t["ms"]))
    nodes = {}
    for name, ms in per_node.items():
        p = percentiles(ms)
        nodes[name] = {"n": len(ms), "p50_ms": p[50], "p95_ms": p[95]}
    totals = [float(r["duration_ms"]) for r in runs if "duration_ms" in r]
    return {
        "runs": len(runs),
        "nodes": nodes,
        "run_ms": percentiles(totals) if totals else {50: 0.0, 95: 0.0},
        "cost_usd_total": round(sum(float(r.get("cost_usd", 0) or 0) for r in runs), 6),
        "tokens_in_total": int(sum(int(r.get("tokens_in", 0) or 0) for r in runs)),
        "tokens_out_total": int(sum(int(r.get("tokens_out", 0) or 0) for r in runs)),
    }
