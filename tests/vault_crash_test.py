"""Crash-safety of enabling/disabling passphrase protection (two files must change together)."""
import pytest

import vault as vault_mod
from vault import KeyVault, VaultError

SECRET = "sk-crashsafe-secret-1234567890"
PASS = "correct horse battery"


class Crash(Exception):
    """Stands in for the process dying at a precise step."""


def _seed(tmp_path):
    v = KeyVault(tmp_path, use_keyring=False)
    v.set("work", "openai", SECRET)
    return v


def _crash_when_replacing(monkeypatch, dst_name):
    real = vault_mod.replace_with_retry

    def flaky(src, dst, attempts=20):
        if str(dst).endswith(dst_name):
            raise Crash(dst_name)
        return real(src, dst, attempts)

    monkeypatch.setattr(vault_mod, "replace_with_retry", flaky)
    return lambda: monkeypatch.setattr(vault_mod, "replace_with_retry", real)


def _leftovers(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.name.endswith((".pending", ".commit")))


# ---------------------------------------------------------------- enable
def test_enable_crash_before_commit_point_rolls_back(tmp_path, monkeypatch):
    v = _seed(tmp_path)
    restore = _crash_when_replacing(monkeypatch, "vault.commit")
    with pytest.raises(Crash):
        v.enable_passphrase(PASS)
    restore()
    reopened = KeyVault(tmp_path, use_keyring=False)
    assert not reopened.uses_passphrase and reopened.get("work") == SECRET   # old state fully intact
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("crash_at", ["vault.enc", "vault.meta"])
def test_enable_crash_after_commit_point_rolls_forward(tmp_path, monkeypatch, crash_at):
    v = _seed(tmp_path)
    restore = _crash_when_replacing(monkeypatch, crash_at)
    with pytest.raises(Crash):
        v.enable_passphrase(PASS)
    restore()
    reopened = KeyVault(tmp_path, use_keyring=False)          # start-up recovery finishes the job
    assert reopened.uses_passphrase and reopened.is_locked
    reopened.unlock(PASS)
    assert reopened.get("work") == SECRET
    with pytest.raises(VaultError):
        KeyVault(tmp_path, use_keyring=False).unlock("wrong passphrase!!")
    assert _leftovers(tmp_path) == []


# ---------------------------------------------------------------- disable
def _protected(tmp_path):
    v = _seed(tmp_path)
    v.enable_passphrase(PASS)
    return v


def test_disable_crash_before_commit_point_rolls_back(tmp_path, monkeypatch):
    v = _protected(tmp_path)
    restore = _crash_when_replacing(monkeypatch, "vault.commit")
    with pytest.raises(Crash):
        v.disable_passphrase(PASS)
    restore()
    reopened = KeyVault(tmp_path, use_keyring=False)
    assert reopened.uses_passphrase
    reopened.unlock(PASS)
    assert reopened.get("work") == SECRET
    assert _leftovers(tmp_path) == []


def test_disable_crash_after_vault_replaced_rolls_forward(tmp_path, monkeypatch):
    """Vault already re-encrypted under the master key but vault.meta still present: must not brick."""
    v = _protected(tmp_path)
    real_unlink = type(tmp_path).unlink

    def crashing_unlink(self, *a, **k):
        if self.name == "vault.meta":
            raise Crash("meta unlink")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "unlink", crashing_unlink)
    with pytest.raises(Crash):
        v.disable_passphrase(PASS)
    monkeypatch.undo()
    reopened = KeyVault(tmp_path, use_keyring=False)
    assert not reopened.uses_passphrase and reopened.get("work") == SECRET
    assert _leftovers(tmp_path) == []


# ---------------------------------------------------------------- normal paths leave no residue
def test_clean_enable_disable_cycle_leaves_no_journal_files(tmp_path):
    v = _seed(tmp_path)
    v.enable_passphrase(PASS)
    assert _leftovers(tmp_path) == []
    v.disable_passphrase(PASS)
    assert _leftovers(tmp_path) == []
    assert KeyVault(tmp_path, use_keyring=False).get("work") == SECRET


def test_recovery_is_idempotent(tmp_path, monkeypatch):
    v = _seed(tmp_path)
    restore = _crash_when_replacing(monkeypatch, "vault.meta")
    with pytest.raises(Crash):
        v.enable_passphrase(PASS)
    restore()
    for _ in range(3):                                    # repeated restarts never corrupt anything
        k = KeyVault(tmp_path, use_keyring=False)
        k.unlock(PASS)
        assert k.get("work") == SECRET
