"""Clients for the language model providers.

Every provider is reached over plain HTTP with ``httpx``, which keeps the
dependency footprint small and makes the clients easy to test. The active
provider is chosen with ``llm.provider`` in the config, so switching between a
local Ollama model and a hosted free tier needs no code change.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import httpx

from resolveiq.config import LLMConfig, ProviderConfig
from resolveiq.rag.prompts import ChatMessage


class LLMError(RuntimeError):
    """Raised when the language model cannot produce an answer."""


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, messages: Sequence[ChatMessage]) -> str: ...


def stream_text(client: LLMClient, messages: Sequence[ChatMessage]) -> Iterator[str]:
    """Yield the answer in pieces; clients without streaming yield it whole."""
    stream = getattr(client, "stream", None)
    if callable(stream):
        yield from stream(messages)
    else:
        yield client.complete(messages)


class _HTTPClient:
    provider = "base"

    def __init__(
        self,
        provider_config: ProviderConfig,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = provider_config.model
        self.base_url = provider_config.base_url.rstrip("/")
        self.api_key_env = provider_config.api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._http = http_client or httpx.Client(timeout=timeout_seconds)

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env or "", "").strip()
        if not key:
            raise LLMError(
                f"{self.provider} needs an API key in the {self.api_key_env} environment variable"
            )
        return key

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            response = self._http.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"{self.provider} returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

    def _stream_lines(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> Iterator[str]:
        """POST and yield the non-empty lines of a streamed response."""
        try:
            with self._http.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", "replace")
                    raise LLMError(
                        f"{self.provider} returned HTTP {response.status_code}: {body[:200]}"
                    )
                for line in response.iter_lines():
                    if line.strip():
                        yield line.strip()
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

    def _sse_payloads(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> Iterator[dict[str, Any]]:
        """Decode server-sent event lines of the form ``data: {...}``."""
        for line in self._stream_lines(url, payload, headers):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except ValueError as exc:
                raise LLMError(f"Unexpected stream format from {self.provider}") from exc


class OllamaClient(_HTTPClient):
    """Local models served by Ollama (``ollama serve``)."""

    provider = "ollama"

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        data = self._post(f"{self.base_url}/api/chat", payload, headers={})
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Unexpected response format from ollama") from exc

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        for line in self._stream_lines(f"{self.base_url}/api/chat", payload, headers={}):
            try:
                chunk = json.loads(line)
            except ValueError as exc:
                raise LLMError("Unexpected stream format from ollama") from exc
            if "error" in chunk:
                raise LLMError(f"ollama error: {str(chunk['error'])[:200]}")
            text = (chunk.get("message") or {}).get("content", "")
            if text:
                yield text
            if chunk.get("done"):
                return


class OpenAICompatibleClient(_HTTPClient):
    """Providers exposing the OpenAI chat completions format (Groq, OpenRouter)."""

    def __init__(self, provider: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.provider = provider

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        data = self._post(f"{self.base_url}/chat/completions", payload, headers)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response format from {self.provider}") from exc

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        for event in self._sse_payloads(f"{self.base_url}/chat/completions", payload, headers):
            try:
                text = (event["choices"][0].get("delta") or {}).get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError):
                continue
            if text:
                yield text


class GeminiClient(_HTTPClient):
    """Google Gemini through the Generative Language REST API."""

    provider = "gemini"

    def _payload(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        data = self._post(url, self._payload(messages), headers={"x-goog-api-key": self._api_key()})
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response format from gemini") from exc

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse"
        headers = {"x-goog-api-key": self._api_key()}
        for event in self._sse_payloads(url, self._payload(messages), headers):
            try:
                parts = event["candidates"][0]["content"]["parts"]
            except (KeyError, IndexError, TypeError):
                continue
            text = "".join(part.get("text", "") for part in parts)
            if text:
                yield text


def create_llm_client(config: LLMConfig, http_client: httpx.Client | None = None) -> LLMClient:
    """Build the client for the provider selected in the config."""
    common = {
        "provider_config": config.active,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "http_client": http_client,
    }
    if config.provider == "ollama":
        return OllamaClient(**common)
    if config.provider == "gemini":
        return GeminiClient(**common)
    return OpenAICompatibleClient(config.provider, **common)
