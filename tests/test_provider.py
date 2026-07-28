import json

import httpx

from recagent_eval.provider import OpenAICompatibleProvider


def test_provider_returns_structured_response_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["temperature"] == 0
        assert payload["response_format"]["type"] == "json_object"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"steps":[{"tool":"rerank","args":{"top_k":10}}]}'}}
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    result = provider.chat(
        [{"role": "user", "content": "recommend movies"}],
        response_schema={"type": "object"},
        timeout=2,
    )

    assert result.structured == {"steps": [{"tool": "rerank", "args": {"top_k": 10}}]}
    assert result.usage.total_tokens == 20
    assert result.error is None
    assert result.latency_ms >= 0


def test_provider_retries_rate_limit_and_does_not_expose_api_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            },
        )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="top-secret",
        model="test-model",
        max_retries=1,
        retry_backoff_seconds=0,
        transport=httpx.MockTransport(handler),
    )

    result = provider.chat([{"role": "user", "content": "hello"}], timeout=2)

    assert calls == 2
    assert result.error is None
    assert "top-secret" not in repr(result)


def test_provider_returns_typed_error_for_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "not-json"}}],
                "usage": {},
            },
        )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    result = provider.chat(
        [{"role": "user", "content": "hello"}],
        response_schema={"type": "object"},
        timeout=2,
    )

    assert result.structured is None
    assert result.error is not None
    assert result.error.code == "invalid_json"
