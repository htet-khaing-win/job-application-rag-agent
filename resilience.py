"""Self-healing primitives: retry with backoff, provider failover, circuit breaker,
JSON output repair, and a per-run healing log."""
from __future__ import annotations

import contextvars
import json
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from security import redact_secrets

# Tunables (tests monkeypatch these to keep runs instant).
MAX_RETRIES_PER_CANDIDATE = 2
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 20.0
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 60.0

_heal_log: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar("heal_log", default=None)


def start_heal_log() -> List[str]:
    log: List[str] = []
    _heal_log.set(log)
    return log


def record_heal(event: str) -> None:
    event = redact_secrets(event)
    log = _heal_log.get()
    if log is not None:
        log.append(event)


def drain_heal_log() -> List[str]:
    """Return and clear events recorded since the last drain."""
    log = _heal_log.get()
    if not log:
        return []
    events = list(log)
    log.clear()
    return events


class AllCandidatesFailed(RuntimeError):
    pass


# --- circuit breaker ------------------------------------------------------
_breaker: Dict[str, Tuple[int, float]] = {}
_breaker_lock = threading.Lock()


def _breaker_open(label: str) -> bool:
    with _breaker_lock:
        fails, since = _breaker.get(label, (0, 0.0))
        if fails >= BREAKER_THRESHOLD:
            if time.monotonic() - since < BREAKER_COOLDOWN_S:
                return True
            _breaker.pop(label, None)  # half-open: allow a probe
        return False


def _breaker_fail(label: str) -> None:
    with _breaker_lock:
        fails, _ = _breaker.get(label, (0, 0.0))
        _breaker[label] = (fails + 1, time.monotonic())


def _breaker_ok(label: str) -> None:
    with _breaker_lock:
        _breaker.pop(label, None)


def reset_breakers() -> None:
    with _breaker_lock:
        _breaker.clear()


_FATAL_HINTS = ("api key", "api_key", "unauthorized", "401", "403", "permission", "invalid_api",
                "authentication", "not installed", "no api key")


def _is_fatal(exc: BaseException) -> bool:
    """Auth/config errors will not fix themselves by retrying the same candidate."""
    msg = str(exc).lower()
    return any(h in msg for h in _FATAL_HINTS)


def _sleep_backoff(attempt: int) -> None:
    delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** attempt))
    time.sleep(delay * (0.5 + random.random() / 2) if delay else 0)


class ResilientLLM:
    """Drop-in `.invoke()` object that retries, then fails over across candidates.

    candidates: ordered [(label, factory)] — factory() builds the underlying chat model
    lazily so a missing key on a fallback never breaks construction of the primary.
    """

    def __init__(self, role: str, candidates: Sequence[Tuple[str, Callable[[], Any]]]):
        if not candidates:
            raise ValueError(f"Role '{role}' has no model candidates configured.")
        self.role = role
        self.candidates = list(candidates)

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        errors: List[str] = []
        for label, factory in self.candidates:
            if _breaker_open(label):
                record_heal(f"[{self.role}] skipped {label}: circuit open")
                errors.append(f"{label}: circuit open")
                continue
            for attempt in range(MAX_RETRIES_PER_CANDIDATE + 1):
                try:
                    result = factory().invoke(prompt, *args, **kwargs)
                    _breaker_ok(label)
                    if errors or attempt:
                        record_heal(f"[{self.role}] recovered on {label} (attempt {attempt + 1})")
                    return result
                except Exception as e:  # noqa: BLE001 - boundary: any provider error
                    msg = redact_secrets(f"{type(e).__name__}: {e}")[:300]
                    _breaker_fail(label)
                    fatal = _is_fatal(e)
                    record_heal(f"[{self.role}] {label} failed (attempt {attempt + 1}): {msg}")
                    if fatal or attempt == MAX_RETRIES_PER_CANDIDATE:
                        errors.append(f"{label}: {msg}")
                        break
                    _sleep_backoff(attempt)
        raise AllCandidatesFailed(f"[{self.role}] all models failed: " + " | ".join(errors))


def _strip_json(text: str) -> str:
    import re
    text = text.strip().replace("```json", "").replace("```", "")
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return re.sub(r" {2,}", " ", text).strip()


def invoke_json(llm: Any, prompt: str, repair_attempts: int = 1,
                validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
                role: str = "llm") -> Dict[str, Any]:
    """Invoke `llm` and parse a JSON object; on failure ask the model to repair it.

    Raises json.JSONDecodeError / ValueError when output cannot be healed so callers
    keep their existing deterministic fallbacks.
    """
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    last_err: Exception = ValueError("empty response")
    for attempt in range(repair_attempts + 1):
        try:
            data = json.loads(_strip_json(content))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            if validator and not validator(data):
                raise ValueError("JSON failed schema validation")
            if attempt:
                record_heal(f"[{role}] repaired malformed JSON on repair attempt {attempt}")
            return data
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            if attempt == repair_attempts:
                break
            repair = (
                "Your previous reply was not valid JSON for the requested schema "
                f"({e}). Return ONLY the corrected JSON object, no prose, no markdown.\n\n"
                f"PREVIOUS REPLY:\n{content[:4000]}\n\nORIGINAL TASK:\n{prompt[:6000]}"
            )
            r = llm.invoke(repair)
            content = r.content if hasattr(r, "content") else str(r)
    record_heal(f"[{role}] JSON could not be repaired: {last_err}")
    if isinstance(last_err, json.JSONDecodeError):
        raise last_err
    raise json.JSONDecodeError(str(last_err), content, 0)
