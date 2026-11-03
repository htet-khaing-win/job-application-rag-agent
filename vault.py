"""Encrypted API-key vault.

Keys are stored as named entries ("aliases") encrypted with Fernet. By default the master key
lives in the OS keyring (Windows Credential Manager / macOS Keychain / Secret Service); if no
keyring backend exists it falls back to a 0600 key file outside the repo.

Optional passphrase mode: the Fernet key is derived from a passphrase with scrypt (salt in
vault.meta) so nothing on disk can decrypt the vault without the passphrase. The vault can then be
locked/unlocked at runtime. Raw keys are never written to .env, logs, or the profile JSON.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import secrets
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken

from security import mask_key, replace_with_retry

APP_DIR = Path(os.environ.get("JOBAGENT_HOME", Path.home() / ".job_application_agent"))
_KEYRING_SERVICE = "job-application-agent"
_KEYRING_USER = "vault-master-key"
_CANARY = b"job-agent-vault-canary"
_LAST_USED_PERSIST_S = 3600
ROTATE_AFTER_DAYS = 90
_MIN_PASSPHRASE = 8


class VaultError(RuntimeError):
    pass


class VaultLocked(VaultError):
    """Raised when the vault is passphrase-protected and currently locked."""


def _restrict(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _load_master_key(app_dir: Path, use_keyring: bool) -> bytes:
    if use_keyring:
        try:
            import keyring

            existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
            if existing:
                return existing.encode()
            new = Fernet.generate_key()
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, new.decode())
            return new
        except Exception:
            pass  # no usable backend -> file fallback
    key_file = app_dir / "master.key"
    if key_file.exists():
        return key_file.read_bytes().strip()
    new = Fernet.generate_key()
    key_file.write_bytes(new)
    _restrict(key_file)
    return new


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    raw = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return base64.urlsafe_b64encode(raw)


def _iso(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def _parse(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        parsed = dt.datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


class KeyVault:
    """Thread-safe encrypted store of {alias: {provider, secret, created_at, last_used, monthly_cap_usd}}."""

    def __init__(self, app_dir: Optional[Path] = None, use_keyring: bool = True,
                 passphrase: Optional[str] = None):
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.app_dir / "vault.enc"
        self._meta_path = self.app_dir / "vault.meta"
        self._use_keyring = use_keyring
        self._lock = threading.RLock()
        self._session: Dict[str, Dict[str, Any]] = {}  # memory-only keys, never persisted
        self._now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)
        self._fernet: Optional[Fernet] = None
        self._pass_mode = self._meta_path.exists()
        if self._pass_mode:
            if passphrase is not None:
                self.unlock(passphrase)
        elif passphrase is not None:
            self._fernet = Fernet(_load_master_key(self.app_dir, use_keyring))
            self.enable_passphrase(passphrase)
        else:
            self._fernet = Fernet(_load_master_key(self.app_dir, use_keyring))

    # -- lock state ----------------------------------------------------------
    @property
    def is_locked(self) -> bool:
        return self._fernet is None

    @property
    def uses_passphrase(self) -> bool:
        return self._pass_mode

    def _f(self) -> Fernet:
        if self._fernet is None:
            raise VaultLocked("Vault is locked. Unlock it with your passphrase first.")
        return self._fernet

    def _read_meta(self) -> Dict[str, str]:
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            salt, canary = meta["salt"], meta["canary"]
            if not isinstance(salt, str) or not isinstance(canary, str):
                raise ValueError
            return {"salt": salt, "canary": canary}
        except (OSError, ValueError, KeyError) as e:
            raise VaultError("Vault metadata is missing or corrupt.") from e

    def _fernet_for(self, passphrase: str) -> Fernet:
        """Derive and verify a passphrase against the stored canary."""
        meta = self._read_meta()
        f = Fernet(_derive_key(passphrase, base64.b64decode(meta["salt"])))
        try:
            if f.decrypt(meta["canary"].encode()) != _CANARY:
                raise InvalidToken
        except InvalidToken as e:
            raise VaultError("Wrong passphrase.") from e
        return f

    def unlock(self, passphrase: str) -> None:
        with self._lock:
            if not self._pass_mode:
                return
            self._fernet = self._fernet_for(passphrase)

    def lock(self) -> None:
        """Drop the key (and any memory-only secrets) from memory."""
        with self._lock:
            if self._pass_mode:
                self._fernet = None
            self._session.clear()

    def enable_passphrase(self, passphrase: str) -> None:
        with self._lock:
            if self._pass_mode:
                raise VaultError("Passphrase protection is already enabled.")
            if len(passphrase or "") < _MIN_PASSPHRASE:
                raise VaultError(f"Passphrase must be at least {_MIN_PASSPHRASE} characters.")
            data = self._read()
            salt = secrets.token_bytes(16)
            new_f = Fernet(_derive_key(passphrase, salt))
            self._write_blob(new_f.encrypt(json.dumps(data).encode()))
            self._write_text(self._meta_path, json.dumps({
                "mode": "passphrase",
                "salt": base64.b64encode(salt).decode(),
                "canary": new_f.encrypt(_CANARY).decode(),
            }))
            self._fernet = new_f
            self._pass_mode = True

    def disable_passphrase(self, passphrase: str) -> None:
        with self._lock:
            if not self._pass_mode:
                raise VaultError("Passphrase protection is not enabled.")
            f = self._fernet_for(passphrase)
            self._fernet = f
            data = self._read()
            master = Fernet(_load_master_key(self.app_dir, self._use_keyring))
            self._write_blob(master.encrypt(json.dumps(data).encode()))
            self._meta_path.unlink(missing_ok=True)
            self._fernet = master
            self._pass_mode = False

    # -- persistence -----------------------------------------------------------
    def _read(self) -> Dict[str, Dict[str, Any]]:
        f = self._f()
        if not self._path.exists():
            return {}
        try:
            data = json.loads(f.decrypt(self._path.read_bytes()))
        except (InvalidToken, ValueError) as e:
            raise VaultError("Vault is corrupt or the master key changed.") from e
        try:
            fallback = _iso(dt.datetime.fromtimestamp(self._path.stat().st_mtime, dt.timezone.utc))
        except OSError:
            fallback = _iso(self._now())
        for entry in data.values():  # old vaults lack metadata: backfill in memory
            entry.setdefault("created_at", fallback)
            entry.setdefault("last_used", None)
            entry.setdefault("monthly_cap_usd", None)
        return data

    def _write_text(self, path: Path, text: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.app_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            _restrict(Path(tmp))
            replace_with_retry(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _write_blob(self, blob: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.app_dir)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(blob)
            _restrict(Path(tmp))
            replace_with_retry(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _write(self, data: Dict[str, Dict[str, Any]]) -> None:
        self._write_blob(self._f().encrypt(json.dumps(data).encode()))

    # -- API ---------------------------------------------------------------------
    @staticmethod
    def _check_alias(alias: str) -> str:
        alias = (alias or "").strip()
        if not alias or len(alias) > 64 or not all(c.isalnum() or c in "-_. " for c in alias):
            raise VaultError("Alias must be 1-64 chars: letters, digits, space, - _ .")
        return alias

    @staticmethod
    def _check_cap(cap: Optional[float]) -> Optional[float]:
        if cap is None:
            return None
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
            raise VaultError("Monthly cap must be a positive number of USD.")
        return float(cap)

    def set(self, alias: str, provider: str, secret: str, persist: bool = True,
            monthly_cap_usd: Optional[float] = None) -> None:
        alias = self._check_alias(alias)
        secret = (secret or "").strip()
        if len(secret) < 8:
            raise VaultError("Key looks too short.")
        cap = self._check_cap(monthly_cap_usd)
        with self._lock:
            self._f()  # raises VaultLocked
            now = _iso(self._now())
            if persist:
                data = self._read()
                prev = data.get(alias) or self._session.get(alias) or {}
                data[alias] = {"provider": provider, "secret": secret, "created_at": now,
                               "last_used": None,
                               "monthly_cap_usd": cap if cap is not None else prev.get("monthly_cap_usd")}
                self._write(data)
                self._session.pop(alias, None)
            else:
                prev = self._session.get(alias) or {}
                self._session[alias] = {"provider": provider, "secret": secret, "created_at": now,
                                        "last_used": None,
                                        "monthly_cap_usd": cap if cap is not None else prev.get("monthly_cap_usd")}

    def set_cap(self, alias: str, usd: Optional[float]) -> None:
        cap = self._check_cap(usd)
        with self._lock:
            self._f()
            if alias in self._session:
                self._session[alias]["monthly_cap_usd"] = cap
                return
            data = self._read()
            if alias not in data:
                raise VaultError(f"No such key: {alias}")
            data[alias]["monthly_cap_usd"] = cap
            self._write(data)

    def get_cap(self, alias: str) -> Optional[float]:
        with self._lock:
            self._f()
            if alias in self._session:
                return self._session[alias].get("monthly_cap_usd")
            entry = self._read().get(alias)
            return entry.get("monthly_cap_usd") if entry else None

    def get(self, alias: str) -> Optional[str]:
        with self._lock:
            self._f()
            now = self._now()
            if alias in self._session:
                self._session[alias]["last_used"] = _iso(now)
                return str(self._session[alias]["secret"])
            data = self._read()
            entry = data.get(alias)
            if not entry:
                return None
            last = _parse(entry.get("last_used"))
            if last is None or (now - last).total_seconds() >= _LAST_USED_PERSIST_S:
                entry["last_used"] = _iso(now)
                try:
                    self._write(data)  # throttled: at most one write per hour per vault
                except OSError:
                    pass
            return str(entry["secret"])

    def delete(self, alias: str) -> bool:
        with self._lock:
            self._f()
            self._session.pop(alias, None)
            data = self._read()
            if alias in data:
                del data[alias]
                self._write(data)
                return True
            return False

    def list(self) -> List[Dict[str, Any]]:
        """Metadata only - never the secret."""
        with self._lock:
            self._f()
            now = self._now()
            rows: List[Dict[str, Any]] = []
            persisted = self._read()
            for source, entries in ((True, persisted), (False, self._session)):
                for alias, e in entries.items():
                    created = _parse(e.get("created_at")) or now
                    age = max(0, (now - created).days)
                    rows.append({
                        "alias": alias, "provider": e["provider"], "masked": mask_key(e["secret"]),
                        "persisted": source, "created_at": e.get("created_at"),
                        "last_used": e.get("last_used"), "age_days": age,
                        "rotate_due": age >= ROTATE_AFTER_DAYS,
                        "monthly_cap_usd": e.get("monthly_cap_usd"),
                    })
            return sorted(rows, key=lambda r: r["alias"])
