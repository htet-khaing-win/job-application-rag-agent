"""Run tracking: every eval run writes a self-describing, comparable JSON report."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

FITTED_BM25 = "utilities/fitted_bm25.json"
_PKGS = ("langchain-core", "langgraph", "pydantic", "pinecone", "pinecone-text", "streamlit", "cryptography")


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def package_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for p in _PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def model_ids(profile: Any) -> Dict[str, List[str]]:
    """{role: ['provider:model', ...]} from a runtime.Profile (no keys, no aliases)."""
    if profile is None:
        return {}
    return {role: [f"{b.provider}:{b.model}" for b in bs] for role, bs in profile.roles.items()}


def bm25_warning(mode: str, namespaces: Sequence[str], path: str = FITTED_BM25) -> Optional[str]:
    """The fitted BM25 encoder (utilities/Fitting.py) is fit on EVERY ingested resume, including the
    ones being evaluated, so its IDF statistics have seen the test corpus."""
    if mode == "cold" or not Path(path).exists() or not namespaces:
        return None
    return ("fitted BM25 encoder was fit on the whole resume corpus, which includes the evaluated resumes "
            f"({len(namespaces)} namespace(s)); hybrid scores may be optimistic. Re-run with --cold-bm25 "
            "or fit the encoder on a held-out split.")


def build_report(kind: str, sections: Mapping[str, Any], metrics: Mapping[str, Mapping[str, Any]], *,
                 seeds: Optional[Mapping[str, Any]] = None, profile: Any = None,
                 index_stats: Union[None, Mapping[str, Any], Callable[[], Mapping[str, Any]]] = None,
                 bm25_mode: str = "fitted", evaluated_namespaces: Sequence[str] = (),
                 cost_usd: float = 0.0, warnings: Sequence[str] = ()) -> Dict[str, Any]:
    """metrics: flat {name: {value, direction: 'higher'|'lower', tolerance?}} - what `regress` gates on."""
    stats: Any = None
    if callable(index_stats):
        try:
            stats = dict(index_stats())
        except Exception as e:  # noqa: BLE001
            stats = {"error": type(e).__name__}
    elif index_stats is not None:
        stats = dict(index_stats)
    warns = list(warnings)
    w = bm25_warning(bm25_mode, evaluated_namespaces)
    if w:
        warns.append(w)
    return {
        "kind": kind, "git_sha": git_sha(),
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "models": model_ids(profile), "index_stats": stats, "seeds": dict(seeds or {}),
        "bm25_mode": bm25_mode, "cost_usd_total": round(float(cost_usd), 6),
        "versions": package_versions(), "warnings": warns,
        "metrics": {k: dict(v) for k, v in metrics.items()}, "sections": dict(sections),
    }


def save_report(report: Mapping[str, Any], directory: str = "evals/reports") -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = d / f"report_{report['kind']}_{stamp}_{str(report['git_sha']).replace('+', '-')}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
