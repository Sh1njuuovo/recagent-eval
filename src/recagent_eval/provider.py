from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ProviderError:
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    structured: dict[str, Any] | None = None
    latency_ms: float = 0.0
    usage: TokenUsage = TokenUsage()
    error: ProviderError | None = None


class LLMProvider(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class ProviderStatus:
    requested: str
    active: str
    model: str
    fallback: bool = False
    message: str = ""


@dataclass(frozen=True)
class ProviderSelection:
    provider: LLMProvider
    status: ProviderStatus


class OpenAICompatibleProvider:
    """Small OpenAI-compatible client for DeepSeek and local vLLM endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider_name: str = "openai-compatible",
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = provider_name
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._api_key = api_key
        self._extra_body = _validated_extra_body(extra_body)
        self._transport = transport

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, model={self.model!r}, "
            f"max_retries={self.max_retries!r}, extra_body=<redacted>)"
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> LLMResponse:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if response_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        payload.update(deepcopy(self._extra_body))

        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(transport=self._transport, timeout=timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=payload,
                    )
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    last_error = ProviderError(
                        code=f"http_{response.status_code}",
                        message=f"provider returned HTTP {response.status_code}",
                        retryable=True,
                    )
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_seconds * (2**attempt))
                        continue
                response.raise_for_status()
                body = response.json()
                text = str(body["choices"][0]["message"].get("content") or "")
                usage_data = body.get("usage") or {}
                usage = TokenUsage(
                    prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
                    completion_tokens=int(usage_data.get("completion_tokens") or 0),
                    total_tokens=int(usage_data.get("total_tokens") or 0),
                )
                structured = None
                if response_schema is not None:
                    try:
                        parsed = json.loads(text)
                        if not isinstance(parsed, dict):
                            raise ValueError("structured response must be a JSON object")
                        structured = parsed
                    except (json.JSONDecodeError, ValueError) as exc:
                        return LLMResponse(
                            text=text,
                            latency_ms=_elapsed_ms(started),
                            usage=usage,
                            error=ProviderError("invalid_json", str(exc), retryable=False),
                        )
                return LLMResponse(
                    text=text,
                    structured=structured,
                    latency_ms=_elapsed_ms(started),
                    usage=usage,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = ProviderError(
                    code="transport_error",
                    message=type(exc).__name__,
                    retryable=True,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
            except (httpx.HTTPStatusError, KeyError, TypeError, ValueError) as exc:
                last_error = ProviderError(
                    code="provider_error",
                    message=str(exc),
                    retryable=False,
                )
                break

        return LLMResponse(
            latency_ms=_elapsed_ms(started),
            error=last_error or ProviderError("unknown", "provider request failed"),
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _validated_extra_body(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("extra_body must be a JSON-serializable mapping")
    protected = {"model", "messages", "response_format", "temperature"} & set(value)
    if protected:
        fields = ", ".join(sorted(str(field) for field in protected))
        raise ValueError(f"extra_body cannot override protected request fields: {fields}")
    try:
        copied = deepcopy(dict(value))
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("extra_body must be a JSON-serializable mapping") from exc
    return copied


class RuleBasedProvider:
    """Offline provider used for deterministic smoke tests and CI."""

    name = "rule-based"
    model = "deterministic-offline"

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> LLMResponse:
        del messages, response_schema, timeout
        plan = {
            "preference_patch": {},
            "steps": [
                {"tool": "lookup", "args": {}},
                {"tool": "hard_filter", "args": {}},
                {"tool": "itemcf_retrieve", "args": {"top_k": 100}},
                {"tool": "semantic_retrieve", "args": {"top_k": 100}},
                {"tool": "rerank", "args": {"top_k": 10}},
                {"tool": "explain", "args": {}},
            ],
        }
        return LLMResponse(
            text=json.dumps(plan),
            structured=plan,
            latency_ms=0.0,
        )


def build_provider(
    name: str,
    *,
    allow_fallback: bool = False,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ProviderSelection:
    """Build a provider, allowing an explicit offline fallback for interactive use."""
    env = os.environ if environ is None else environ
    normalized = name.strip().lower()
    supported = {"rule-based", "deepseek", "vllm", "qwen"}
    if normalized not in supported:
        raise ValueError("provider must be rule-based, deepseek, vllm, or qwen")
    if normalized == "rule-based":
        return ProviderSelection(
            RuleBasedProvider(),
            ProviderStatus(normalized, "rule-based", "deterministic-offline"),
        )
    try:
        if normalized == "deepseek":
            key = _required_env(env, "DEEPSEEK_API_KEY")
            model = env.get("DEEPSEEK_MODEL", "deepseek-chat")
            provider = OpenAICompatibleProvider(
                base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                api_key=key,
                model=model,
                provider_name="deepseek",
                transport=transport,
            )
            return ProviderSelection(
                provider,
                ProviderStatus(normalized, "deepseek", model),
            )
        if normalized in {"vllm", "qwen"}:
            base_url = _required_env(env, "VLLM_BASE_URL")
            key = _required_env(env, "VLLM_API_KEY")
            model = env.get("VLLM_MODEL", "Qwen/Qwen3-8B")
            provider = OpenAICompatibleProvider(
                base_url=base_url,
                api_key=key,
                model=model,
                provider_name="vllm/qwen",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                transport=transport,
            )
            return ProviderSelection(
                provider,
                ProviderStatus(normalized, "vllm/qwen", model),
            )
        raise AssertionError("unreachable provider selection")
    except ValueError as exc:
        if not allow_fallback:
            raise
        return ProviderSelection(
            RuleBasedProvider(),
            ProviderStatus(
                normalized,
                "rule-based",
                "deterministic-offline",
                fallback=True,
                message=f"{normalized} unavailable: {exc}; using rule-based provider",
            ),
        )


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if not value:
        raise ValueError(f"{name} is required for this provider")
    return value
