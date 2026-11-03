"""Encrypted API-key vault.

Keys are stored as named entries ("aliases") encrypted with Fernet. The master key
lives in the OS keyring (Windows Credential Manager / macOS Keychain / Secret Service);
if no keyring backend exists it falls back to a 0600 key file outside the repo.
Raw keys are never written to .env, logs, or the profile JSON.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken

from security import mask_key

APP_DIR = Path(os.environ.get("JOBAGENT_HOME", Path.home() / ".job_application_agent"))
_KEYRING_SERVICE = "job-application-agent"
_KEYRING_USER = "vault-master-key"


class VaultError(RuntimeError):
    pass


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


class KeyVault:
    """Thread-safe encrypted store of {alias: {provider, secret}}."""

    def __init__(self, app_dir: Optional[Path] = None, use_keyring: bool = True):
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.app_dir / "vault.enc"
        self._lock = threading.RLock()
        self._fernet = Fernet(_load_master_key(self.app_dir, use_keyring))
        self._session: Dict[str, Dict[str, str]] = {}  # memory-only keys, never persisted

    # -- persistence -------------------------------------------------------
    def _read(self) -> Dict[str, Dict[str, str]]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._fernet.decrypt(self._path.read_bytes()))
        except (InvalidToken, ValueError) as e:
            raise VaultError("Vault is corrupt or the master key changed.") from e

    def _write(self, data: Dict[str, Dict[str, str]]) -> None:
        blob = self._fernet.encrypt(json.dumps(data).encode())
        fd, tmp = tempfile.mkstemp(dir=self.app_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            _restrict(Path(tmp))
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # -- API ---------------------------------------------------------------
    @staticmethod
    def _check_alias(alias: str) -> str:
        alias = (alias or "").strip()
        if not alias or len(alias) > 64 or not all(c.isalnum() or c in "-_. " for c in alias):
            raise VaultError("Alias must be 1-64 chars: letters, digits, space, - _ .")
        return alias

    def set(self, alias: str, provider: str, secret: str, persist: bool = True) -> None:
        alias = self._check_alias(alias)
        secret = (secret or "").strip()
        if len(secret) < 8:
            raise VaultError("Key looks too short.")
        with self._lock:
            if persist:
                data = self._read()
                data[alias] = {"provider": provider, "secret": secret}
                self._write(data)
                self._session.pop(alias, None)
            else:
                self._session[alias] = {"provider": provider, "secret": secret}

    def get(self, alias: str) -> Optional[str]:
        with self._lock:
            if alias in self._session:
                return self._session[alias]["secret"]
            entry = self._read().get(alias)
            return entry["secret"] if entry else None

    def delete(self, alias: str) -> bool:
        with self._lock:
            self._session.pop(alias, None)
            data = self._read()
            if alias in data:
                del data[alias]
                self._write(data)
                return True
            return False

    def list(self) -> List[Dict[str, str]]:
        """Metadata only: alias, provider, masked tail, persisted flag. Never the secret."""
        with self._lock:
            out = [
                {"alias": a, "provider": e["provider"], "masked": mask_key(e["secret"]), "persisted": True}
                for a, e in self._read().items()
            ]
            out += [
                {"alias": a, "provider": e["provider"], "masked": mask_key(e["secret"]), "persisted": False}
                for a, e in self._session.items()
            ]
            return sorted(out, key=lambda r: r["alias"])
