"""Provider abstraction for LLM-backed concept extraction."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from labelgen.config import LLMExtractionConfig, LLMProviderName

_DEFAULT_BASE_URLS: dict[LLMProviderName, str] = {
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
_DEFAULT_API_KEY_ENV_VARS: dict[LLMProviderName, str] = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}


class LLMProviderClient(ABC):
    """Abstract provider client for one chat-completion style request."""

    @abstractmethod
    def complete_chat(
        self,
        *,
        messages: list[dict[str, str]],
        config: LLMExtractionConfig,
    ) -> str:
        """Return the message content for one structured extraction request."""


class OpenAICompatibleProviderClient(LLMProviderClient):
    """Chat-completions client for OpenAI-compatible providers."""

    def complete_chat(
        self,
        *,
        messages: list[dict[str, str]],
        config: LLMExtractionConfig,
    ) -> str:
        """Send a chat completion request and return the assistant content."""

        api_key = self._resolve_api_key(config)
        url = self._resolve_chat_completions_url(config)
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if config.organization:
            headers["OpenAI-Organization"] = config.organization

        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            try:
                response = self._post_json(url, headers, payload, timeout=config.timeout_seconds)
                return self._extract_content(response)
            except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
                last_error = error
                if attempt >= config.max_retries:
                    break
                time.sleep(min(2**attempt, 5))
        raise RuntimeError("LLM provider request failed after retries.") from last_error

    def _resolve_api_key(self, config: LLMExtractionConfig) -> str:
        """Resolve the provider API key from explicit config or environment."""

        env_var = config.api_key_env_var or _DEFAULT_API_KEY_ENV_VARS[config.provider]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{config.provider}'. Set environment variable "
                f"'{env_var}' or configure `api_key_env_var`."
            )
        return api_key

    def _resolve_chat_completions_url(self, config: LLMExtractionConfig) -> str:
        """Resolve the provider chat-completions endpoint URL."""

        base_url = config.base_url or _DEFAULT_BASE_URLS[config.provider]
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        """Post JSON to a provider endpoint and return the decoded JSON object."""

        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise RuntimeError("LLM provider response must be a JSON object.")
        return cast(dict[str, Any], data)

    def _extract_content(self, response: dict[str, Any]) -> str:
        """Extract assistant message content from a chat-completions response."""

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM provider response is missing choices.")
        first_choice = cast(list[object], choices)[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("LLM provider choice must be a JSON object.")
        choice = cast(dict[str, Any], first_choice)
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM provider response is missing message content.")
        message_dict = cast(dict[str, Any], message)
        content = message_dict.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in cast(list[object], content):
                if not isinstance(item, dict):
                    continue
                item_dict = cast(dict[str, Any], item)
                text = item_dict.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
        raise RuntimeError("LLM provider response content must be textual.")


def build_provider_client(config: LLMExtractionConfig) -> LLMProviderClient:
    """Build the provider client for the configured provider."""

    if config.provider in {"openai", "mistral", "qwen"}:
        return OpenAICompatibleProviderClient()
    raise RuntimeError(f"Unsupported LLM provider '{config.provider}'.")
