import json
from collections import UserDict

import httpx

from recagent_eval.provider import (
    OpenAICompatibleProvider,
    RuleBasedProvider,
    build_provider,
)


def _success_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "{}"}}], "usage": {}},
    )


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


def test_provider_rejects_structured_arrays_and_malformed_success_payloads() -> None:
    array_provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "[]"}}]},
            )
        ),
    )
    array_response = array_provider.chat(
        [{"role": "user", "content": "hello"}], response_schema={"type": "object"}
    )

    malformed_provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"usage": {}})
        ),
    )
    malformed_response = malformed_provider.chat(
        [{"role": "user", "content": "hello"}]
    )

    assert array_response.error is not None
    assert array_response.error.code == "invalid_json"
    assert "JSON object" in array_response.error.message
    assert malformed_response.error is not None
    assert malformed_response.error.code == "provider_error"
    assert malformed_response.error.retryable is False


def test_provider_deep_copies_and_forwards_extra_body() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _success_response()

    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
        extra_body=extra,
        transport=httpx.MockTransport(handler),
    )
    extra["chat_template_kwargs"]["enable_thinking"] = True

    provider.chat([{"role": "user", "content": "hello"}])

    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_provider_rejects_protected_extra_body_fields() -> None:
    for field in ("model", "messages", "response_format", "temperature"):
        try:
            OpenAICompatibleProvider(
                base_url="https://example.test/v1",
                api_key="secret",
                model="test-model",
                extra_body={field: "override"},
            )
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"protected field {field} was accepted")


def test_provider_rejects_non_mapping_or_non_json_extra_body_without_leaking_value() -> None:
    secret = "do-not-leak-extra-body-secret"
    invalid_values = [UserDict({"value": object()}), [secret]]

    for invalid in invalid_values:
        try:
            OpenAICompatibleProvider(
                base_url="https://example.test/v1",
                api_key="api-secret",
                model="test-model",
                extra_body=invalid,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            assert secret not in str(exc)
            assert "api-secret" not in str(exc)
        else:
            raise AssertionError("invalid extra_body was accepted")


def test_provider_repr_redacts_api_key_and_extra_body() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="api-secret",
        model="test-model",
        extra_body={"vendor_token": "body-secret"},
    )

    rendered = repr(provider)

    assert "api-secret" not in rendered
    assert "body-secret" not in rendered


def test_qwen_provider_factory_sets_non_thinking_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _success_response()

    selection = build_provider(
        "qwen",
        environ={
            "VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
            "VLLM_API_KEY": "secret",
        },
        transport=httpx.MockTransport(handler),
    )

    selection.provider.chat([{"role": "user", "content": "hello"}])

    assert captured["model"] == "Qwen/Qwen3-8B"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert selection.status.requested == "qwen"
    assert selection.status.active == "vllm/qwen"
    assert selection.status.fallback is False


def test_deepseek_provider_factory_uses_configured_endpoint_model_and_key() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured.update(json.loads(request.content))
        return _success_response()

    selection = build_provider(
        " DeepSeek ",
        environ={
            "DEEPSEEK_API_KEY": "configured-secret",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1/",
            "DEEPSEEK_MODEL": "deepseek-reasoner",
        },
        transport=httpx.MockTransport(handler),
    )
    response = selection.provider.chat([{"role": "user", "content": "hello"}])

    assert response.error is None
    assert captured == {
        "url": "https://deepseek.example/v1/chat/completions",
        "authorization": "Bearer configured-secret",
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }
    assert selection.status.active == "deepseek"
    assert selection.status.model == "deepseek-reasoner"


def test_demo_provider_factory_falls_back_visibly_when_configuration_missing() -> None:
    for requested in ("deepseek", "vllm", "qwen"):
        selection = build_provider(requested, environ={}, allow_fallback=True)

        assert isinstance(selection.provider, RuleBasedProvider)
        assert selection.status.active == "rule-based"
        assert selection.status.fallback is True
        assert selection.status.message


def test_formal_provider_factory_rejects_missing_remote_configuration() -> None:
    for requested, expected in (
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("vllm", "VLLM_BASE_URL"),
    ):
        try:
            build_provider(requested, environ={}, allow_fallback=False)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"missing configuration accepted for {requested}")


def test_provider_factory_never_hides_unknown_provider_name() -> None:
    try:
        build_provider("typo", environ={}, allow_fallback=True)
    except ValueError as exc:
        assert "rule-based, deepseek, vllm, or qwen" in str(exc)
    else:
        raise AssertionError("unknown provider silently fell back")
