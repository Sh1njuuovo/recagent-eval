from __future__ import annotations

import json
import time
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


class OpenAICompatibleProvider:
    """Small OpenAI-compatible client for DeepSeek and local vLLM endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._api_key = api_key
        self._transport = transport

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


class RuleBasedProvider:
    """Offline provider used for deterministic smoke tests and CI."""

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
