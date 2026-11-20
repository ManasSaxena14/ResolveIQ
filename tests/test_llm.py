import json

import httpx
import pytest

from resolveiq.config import LLMConfig, load_config
from resolveiq.rag.llm import (
    GeminiClient,
    LLMError,
    OllamaClient,
    OpenAICompatibleClient,
    create_llm_client,
)

MESSAGES = [
    {"role": "system", "content": "Be helpful."},
    {"role": "user", "content": "Where is my order?"},
]


def _config(provider: str) -> LLMConfig:
    config = load_config().llm.model_copy(deep=True)
    config.provider = provider
    return config


def _client_returning(payload: dict, captured: list, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_request_and_response():
    captured: list[httpx.Request] = []
    http = _client_returning({"message": {"content": "It has shipped."}}, captured)

    client = create_llm_client(_config("ollama"), http_client=http)

    assert isinstance(client, OllamaClient)
    assert client.complete(MESSAGES) == "It has shipped."
    body = json.loads(captured[0].content)
    assert captured[0].url.path == "/api/chat"
    assert body["stream"] is False
    assert body["messages"] == MESSAGES
    assert body["options"]["temperature"] == pytest.approx(0.1)


@pytest.mark.parametrize("provider", ["groq", "openrouter"])
def test_openai_compatible_providers(provider, monkeypatch):
    config = _config(provider)
    monkeypatch.setenv(config.active.api_key_env, "test-key")
    captured: list[httpx.Request] = []
    http = _client_returning({"choices": [{"message": {"content": "Hello"}}]}, captured)

    client = create_llm_client(config, http_client=http)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == provider
    assert client.complete(MESSAGES) == "Hello"
    assert captured[0].url.path.endswith("/chat/completions")
    assert captured[0].headers["Authorization"] == "Bearer test-key"
    assert json.loads(captured[0].content)["max_tokens"] == config.max_tokens


def test_gemini_moves_system_prompt_to_system_instruction(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured: list[httpx.Request] = []
    reply = {"candidates": [{"content": {"parts": [{"text": "Part one. "}, {"text": "Two."}]}}]}
    http = _client_returning(reply, captured)

    client = create_llm_client(_config("gemini"), http_client=http)

    assert isinstance(client, GeminiClient)
    assert client.complete(MESSAGES) == "Part one. Two."
    body = json.loads(captured[0].content)
    assert body["systemInstruction"]["parts"][0]["text"] == "Be helpful."
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Where is my order?"}]}]
    assert captured[0].url.path.endswith(":generateContent")
    assert captured[0].headers["x-goog-api-key"] == "test-key"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = create_llm_client(_config("groq"), http_client=_client_returning({}, []))

    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        client.complete(MESSAGES)


def test_http_error_is_wrapped():
    http = _client_returning({"error": "model not found"}, [], status=404)
    client = create_llm_client(_config("ollama"), http_client=http)

    with pytest.raises(LLMError, match="HTTP 404"):
        client.complete(MESSAGES)


def test_connection_error_is_wrapped():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = create_llm_client(_config("ollama"), http_client=http)

    with pytest.raises(LLMError, match="request failed"):
        client.complete(MESSAGES)


def test_unexpected_payload_is_reported(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    http = _client_returning({"choices": []}, [])
    client = create_llm_client(_config("openrouter"), http_client=http)

    with pytest.raises(LLMError, match="Unexpected response"):
        client.complete(MESSAGES)


def _client_streaming(body: str, captured: list, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, content=body.encode())

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_streams_ndjson(monkeypatch):
    captured: list[httpx.Request] = []
    body = "\n".join(
        json.dumps(chunk)
        for chunk in [
            {"message": {"content": "It has "}, "done": False},
            {"message": {"content": "shipped."}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
    )
    client = create_llm_client(_config("ollama"), http_client=_client_streaming(body, captured))

    assert list(client.stream(MESSAGES)) == ["It has ", "shipped."]
    assert json.loads(captured[0].content)["stream"] is True


def test_openai_compatible_streams_sse(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    captured: list[httpx.Request] = []
    body = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Refunds take "}}]}\n\n'
        ": keep-alive\n\n"
        'data: {"choices":[{"delta":{"content":"5 days."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    client = create_llm_client(_config("groq"), http_client=_client_streaming(body, captured))

    assert "".join(client.stream(MESSAGES)) == "Refunds take 5 days."
    assert captured[0].headers["Authorization"] == "Bearer test-key"


def test_gemini_streams_sse(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    captured: list[httpx.Request] = []
    body = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}\n\n'
        'data: {"candidates":[{"content":{"parts":[{"text":"there"}]}}]}\n\n'
    )
    client = create_llm_client(_config("gemini"), http_client=_client_streaming(body, captured))

    assert list(client.stream(MESSAGES)) == ["Hello ", "there"]
    assert "streamGenerateContent" in str(captured[0].url)


def test_stream_errors_become_llm_errors(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    failing = create_llm_client(
        _config("groq"), http_client=_client_streaming("rate limited", [], status=429)
    )
    broken = create_llm_client(
        _config("ollama"), http_client=_client_streaming('{"error": "model not found"}', [])
    )

    with pytest.raises(LLMError, match="HTTP 429"):
        list(failing.stream(MESSAGES))
    with pytest.raises(LLMError, match="model not found"):
        list(broken.stream(MESSAGES))


def test_stream_text_falls_back_to_complete():
    from resolveiq.rag.llm import stream_text

    class Plain:
        provider = "plain"
        model = "plain"

        def complete(self, messages):
            return "whole answer"

    assert list(stream_text(Plain(), MESSAGES)) == ["whole answer"]
