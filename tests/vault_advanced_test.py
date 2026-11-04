import datetime as dt
import json

import pytest
from cryptography.fernet import Fernet

from vault import KeyVault, VaultError, VaultLocked

PW = "correct horse battery"


def mk(tmp_path, **kw):
    return KeyVault(tmp_path, use_keyring=False, **kw)


def clock(v, when):
    v._now = lambda: when


T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


# --- passphrase mode ---------------------------------------------------------
def test_enable_passphrase_reencrypts_and_requires_it_afterwards(tmp_path):
    v = mk(tmp_path)
    v.set("a", "openai", "sk-secret-value-123456")
    v.enable_passphrase(PW)
    assert (tmp_path / "vault.meta").exists()
    assert b"secret-value" not in (tmp_path / "vault.enc").read_bytes()
    assert v.get("a") == "sk-secret-value-123456"

    reopened = mk(tmp_path)                       # no passphrase -> locked
    assert reopened.is_locked and reopened.uses_passphrase
    with pytest.raises(VaultLocked):
        reopened.get("a")
    reopened.unlock(PW)
    assert not reopened.is_locked and reopened.get("a") == "sk-secret-value-123456"


def test_wrong_passphrase_rejected_and_stays_locked(tmp_path):
    v = mk(tmp_path)
    v.set("a", "openai", "sk-secret-value-123456")
    v.enable_passphrase(PW)
    v.lock()
    with pytest.raises(VaultError, match="Wrong passphrase"):
        v.unlock("not the passphrase")
    assert v.is_locked
    with pytest.raises(VaultLocked):
        v.list()


def test_lock_blocks_every_operation_and_drops_session_keys(tmp_path):
    v = mk(tmp_path, passphrase=PW)               # fresh vault + passphrase => passphrase mode
    v.set("keep", "openai", "sk-persisted-0000000")
    v.set("tmp", "openai", "sk-session-only-0000", persist=False)
    v.lock()
    for op in (lambda: v.get("keep"), lambda: v.list(), lambda: v.delete("keep"),
               lambda: v.set("x", "openai", "sk-1234567890")):
        with pytest.raises(VaultLocked):
            op()
    v.unlock(PW)
    assert v.get("keep") and v.get("tmp") is None  # memory-only secret did not survive the lock


def test_constructor_passphrase_unlocks_existing_vault(tmp_path):
    mk(tmp_path, passphrase=PW).set("a", "openai", "sk-secret-value-123456")
    assert mk(tmp_path, passphrase=PW).get("a") == "sk-secret-value-123456"
    with pytest.raises(VaultError):
        mk(tmp_path, passphrase="wrong wrong wrong")


def test_short_passphrase_and_double_enable_rejected(tmp_path):
    v = mk(tmp_path)
    with pytest.raises(VaultError):
        v.enable_passphrase("short")
    v.enable_passphrase(PW)
    with pytest.raises(VaultError):
        v.enable_passphrase(PW)


def test_disable_passphrase_returns_to_master_key_mode(tmp_path):
    v = mk(tmp_path, passphrase=PW)
    v.set("a", "openai", "sk-secret-value-123456")
    with pytest.raises(VaultError):
        v.disable_passphrase("wrong wrong wrong")
    v.disable_passphrase(PW)
    assert not (tmp_path / "vault.meta").exists()
    again = mk(tmp_path)
    assert not again.is_locked and again.get("a") == "sk-secret-value-123456"


def test_passphrase_is_not_stored_anywhere(tmp_path):
    mk(tmp_path, passphrase=PW).set("a", "openai", "sk-secret-value-123456")
    for p in tmp_path.iterdir():
        assert PW.encode() not in p.read_bytes()


# --- backward compatibility -------------------------------------------------
def test_old_format_vault_loads_and_gains_metadata(tmp_path):
    v = mk(tmp_path)
    old = {"legacy": {"provider": "openai", "secret": "sk-legacy-value-123456"}}   # pre-metadata shape
    (tmp_path / "vault.enc").write_bytes(Fernet((tmp_path / "master.key").read_bytes().strip())
                                         .encrypt(json.dumps(old).encode()))
    assert v.get("legacy") == "sk-legacy-value-123456"
    row = v.list()[0]
    assert row["alias"] == "legacy" and row["created_at"] and row["monthly_cap_usd"] is None
    assert "secret" not in row and "legacy-value" not in str(row)


# --- last_used throttling ---------------------------------------------------
def test_last_used_persisted_at_most_once_per_hour(tmp_path):
    v = mk(tmp_path)
    clock(v, T0)
    v.set("a", "openai", "sk-secret-value-123456")
    assert v.list()[0]["last_used"] is None
    v.get("a")
    first = (tmp_path / "vault.enc").read_bytes()
    assert v.list()[0]["last_used"] == "2026-01-01T00:00:00+00:00"

    clock(v, T0 + dt.timedelta(minutes=30))
    v.get("a")
    assert (tmp_path / "vault.enc").read_bytes() == first          # no rewrite inside the hour

    clock(v, T0 + dt.timedelta(minutes=61))
    v.get("a")
    assert (tmp_path / "vault.enc").read_bytes() != first
    assert v.list()[0]["last_used"] == "2026-01-01T01:01:00+00:00"


# --- rotation ---------------------------------------------------------------
def test_rotate_due_after_90_days_and_reset_by_replacing_key(tmp_path):
    v = mk(tmp_path)
    clock(v, T0)
    v.set("a", "openai", "sk-secret-value-123456")
    clock(v, T0 + dt.timedelta(days=89))
    assert not v.list()[0]["rotate_due"] and v.list()[0]["age_days"] == 89
    clock(v, T0 + dt.timedelta(days=90))
    assert v.list()[0]["rotate_due"]
    v.set("a", "openai", "sk-rotated-value-98765")                   # rotating resets the clock
    row = v.list()[0]
    assert row["age_days"] == 0 and not row["rotate_due"]


# --- caps -------------------------------------------------------------------
def test_caps_set_update_keep_on_rotation_and_validate(tmp_path):
    v = mk(tmp_path)
    v.set("a", "openai", "sk-secret-value-123456", monthly_cap_usd=5)
    assert v.get_cap("a") == 5.0
    v.set_cap("a", 12.5)
    assert v.list()[0]["monthly_cap_usd"] == 12.5
    v.set("a", "openai", "sk-rotated-value-98765")                   # cap survives key rotation
    assert v.get_cap("a") == 12.5
    v.set_cap("a", None)
    assert v.get_cap("a") is None
    for bad in (0, -1, True, "5"):
        with pytest.raises(VaultError):
            v.set_cap("a", bad)
    with pytest.raises(VaultError):
        v.set_cap("nope", 1)


def test_session_key_caps_and_metadata(tmp_path):
    v = mk(tmp_path)
    v.set("s", "openai", "sk-session-only-0000", persist=False, monthly_cap_usd=2)
    row = v.list()[0]
    assert row["persisted"] is False and row["monthly_cap_usd"] == 2.0
    v.set_cap("s", 3)
    assert v.get_cap("s") == 3.0


def test_listing_never_leaks_secret(tmp_path):
    v = mk(tmp_path, passphrase=PW)
    v.set("a", "openai", "sk-verysecret-0123456789", monthly_cap_usd=1)
    assert "verysecret" not in json.dumps(v.list(), default=str)
