import pytest

from runtime import Binding, Profile, Runtime
from vault import KeyVault, VaultError


@pytest.fixture
def vault(tmp_path):
    return KeyVault(tmp_path, use_keyring=False)


def test_roundtrip_and_encrypted_at_rest(vault, tmp_path):
    vault.set("work", "openai", "sk-supersecret-1234567890")
    assert vault.get("work") == "sk-supersecret-1234567890"
    assert b"supersecret" not in (tmp_path / "vault.enc").read_bytes()


def test_list_never_exposes_secret(vault):
    vault.set("work", "openai", "sk-supersecret-1234567890")
    dumped = str(vault.list())
    assert "supersecret" not in dumped and "7890" in dumped


def test_survives_reopen(tmp_path):
    KeyVault(tmp_path, use_keyring=False).set("a", "gemini", "AIza-key-value-12345")
    assert KeyVault(tmp_path, use_keyring=False).get("a") == "AIza-key-value-12345"


def test_session_keys_not_persisted(vault, tmp_path):
    vault.set("tmp", "openai", "sk-session-only-123456", persist=False)
    assert vault.get("tmp")
    assert KeyVault(tmp_path, use_keyring=False).get("tmp") is None


def test_delete_and_validation(vault):
    vault.set("a", "openai", "sk-12345678901234")
    assert vault.delete("a") and vault.get("a") is None
    with pytest.raises(VaultError):
        vault.set("bad/alias", "openai", "sk-12345678901234")
    with pytest.raises(VaultError):
        vault.set("ok", "openai", "short")


def test_wrong_master_key_fails_closed(tmp_path):
    KeyVault(tmp_path, use_keyring=False).set("a", "openai", "sk-12345678901234")
    (tmp_path / "master.key").unlink()
    with pytest.raises(VaultError):
        KeyVault(tmp_path, use_keyring=False).get("a")


def test_runtime_resolves_alias_then_provider_default(vault, monkeypatch):
    monkeypatch.delenv("JOBAGENT_ALLOW_ENV_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-0000000000")
    rt = Runtime(vault, Profile())
    assert rt.resolve_key(Binding("openai", "m")) is None  # env ignored unless opted in
    vault.set("one", "openai", "sk-one-0000000000")
    vault.set("two", "openai", "sk-two-0000000000")
    assert rt.resolve_key(Binding("openai", "m", key_alias="two")) == "sk-two-0000000000"
    assert rt.resolve_key(Binding("openai", "m")) in ("sk-one-0000000000", "sk-two-0000000000")


def test_env_fallback_requires_explicit_opt_in(vault, monkeypatch):
    monkeypatch.setenv("JOBAGENT_ALLOW_ENV_KEYS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-0000000000")
    assert Runtime(vault, Profile()).resolve_key(Binding("openai", "m")) == "sk-from-env-0000000000"


def test_profile_rejects_bad_binding():
    with pytest.raises(ValueError):
        Profile({"writer": [Binding("tavily", "x")]}).validate()
    with pytest.raises(ValueError):
        Profile({"nope": [Binding("openai", "x")]}).validate()


def test_profile_roundtrip(tmp_path):
    p = Profile()
    p.roles["writer"] = [Binding("anthropic", "claude-sonnet-5-5", "mine")]
    p.save(tmp_path / "p.json")
    assert "sk-" not in (tmp_path / "p.json").read_text()
    assert Profile.load(tmp_path / "p.json").roles["writer"][0].key_alias == "mine"
