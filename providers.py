"""Provider registry and lazy client factories.

Nothing here reads os.environ for credentials: keys are always passed in explicitly
(resolved from the vault by runtime.py). Provider SDKs are imported lazily so a missing
optional SDK only breaks the bindings that use it.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    kinds: tuple          # subset of {"chat", "embedding", "search", "vector"}
    needs_key: bool
    models: Dict[str, List[str]] = field(default_factory=dict)  # kind -> suggested models


PROVIDERS: Dict[str, ProviderSpec] = {
    "gemini": ProviderSpec("gemini", "Google Gemini", ("chat", "embedding"), True, {
        "chat": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "embedding": ["models/text-embedding-004"],
    }),
    "openai": ProviderSpec("openai", "OpenAI", ("chat", "embedding"), True, {
        "chat": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        "embedding": ["text-embedding-3-small"],
    }),
    "anthropic": ProviderSpec("anthropic", "Anthropic Claude", ("chat",), True, {
        "chat": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"],
    }),
    "ollama": ProviderSpec("ollama", "Ollama (local)", ("chat", "embedding"), False, {
        "chat": ["mistral:7b-instruct", "qwen2.5:7b", "llama3.1:8b"],
        "embedding": ["nomic-embed-text"],
    }),
    "tavily": ProviderSpec("tavily", "Tavily Search", ("search",), True, {"search": ["advanced"]}),
    "pinecone": ProviderSpec("pinecone", "Pinecone", ("vector",), True, {"vector": ["resume-index"]}),
}


def providers_for(kind: str) -> List[str]:
    return [n for n, p in PROVIDERS.items() if kind in p.kinds]


class ProviderError(RuntimeError):
    pass


def _fingerprint(*parts: Optional[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode())
        h.update(b"\x00")
    return h.hexdigest()[:24]


_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def _cached(key: str, factory: Callable[[], Any]) -> Any:
    with _cache_lock:
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


def clear_client_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _require_key(spec: ProviderSpec, key: Optional[str]) -> None:
    if spec.needs_key and not key:
        raise ProviderError(f"No API key available for provider '{spec.name}'.")


def build_chat_model(provider: str, model: str, key: Optional[str], temperature: float = 0.0) -> Any:
    spec = PROVIDERS.get(provider)
    if not spec or "chat" not in spec.kinds:
        raise ProviderError(f"Provider '{provider}' does not support chat.")
    _require_key(spec, key)

    def make() -> Any:
        try:
            if provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                gkw: Dict[str, Any] = dict(model=model, temperature=temperature, api_key=key, max_retries=0, timeout=90)
                return ChatGoogleGenerativeAI(**gkw)
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                okw: Dict[str, Any] = dict(model=model, temperature=temperature, api_key=key, max_retries=0, timeout=90)
                return ChatOpenAI(**okw)
            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                akw: Dict[str, Any] = dict(model=model, temperature=temperature, api_key=key, max_retries=0, timeout=90)
                return ChatAnthropic(**akw)
            if provider == "ollama":
                from langchain_ollama import ChatOllama
                return ChatOllama(model=model, temperature=temperature)
        except ImportError as e:
            raise ProviderError(f"SDK for '{provider}' is not installed: {e.name}") from e
        raise ProviderError(f"Unknown provider '{provider}'.")

    return _cached("chat:" + _fingerprint(provider, model, key, str(temperature)), make)


def build_embeddings(provider: str, model: str, key: Optional[str]) -> Any:
    spec = PROVIDERS.get(provider)
    if not spec or "embedding" not in spec.kinds:
        raise ProviderError(f"Provider '{provider}' does not support embeddings.")
    _require_key(spec, key)

    def make() -> Any:
        try:
            if provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                oekw: Dict[str, Any] = dict(model=model, api_key=key)
                return OpenAIEmbeddings(**oekw)
            if provider == "gemini":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                gekw: Dict[str, Any] = dict(model=model, google_api_key=key)
                return GoogleGenerativeAIEmbeddings(**gekw)
            if provider == "ollama":
                from langchain_ollama import OllamaEmbeddings
                return OllamaEmbeddings(model=model)
        except ImportError as e:
            raise ProviderError(f"SDK for '{provider}' is not installed: {e.name}") from e
        raise ProviderError(f"Unknown provider '{provider}'.")

    return _cached("emb:" + _fingerprint(provider, model, key), make)


def build_tavily(key: Optional[str]) -> Any:
    _require_key(PROVIDERS["tavily"], key)

    def make() -> Any:
        try:
            from tavily import TavilyClient
        except ImportError as e:
            raise ProviderError("tavily-python is not installed.") from e
        return TavilyClient(api_key=key)

    return _cached("tavily:" + _fingerprint(key), make)


def build_pinecone(key: Optional[str]) -> Any:
    _require_key(PROVIDERS["pinecone"], key)

    def make() -> Any:
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise ProviderError("pinecone is not installed.") from e
        return Pinecone(api_key=key)

    return _cached("pinecone:" + _fingerprint(key), make)
