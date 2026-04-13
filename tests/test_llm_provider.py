"""Tests for the OpenAI-compatible LLM provider client."""

from __future__ import annotations

from email.message import Message
from typing import Any, cast
from urllib.error import HTTPError, URLError

from labelgen import LabelGeneratorConfig
from labelgen.extraction.llm_provider import (
    LLMProviderConfigurationError,
    LLMProviderHTTPStatusError,
    LLMProviderRetryExhaustedError,
    LLMProviderTransportError,
    OpenAICompatibleProviderClient,
)


class RecordingProviderClient(OpenAICompatibleProviderClient):
    """Provider client that records request payloads instead of making HTTP calls."""

    def __init__(self) -> None:
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_payload: dict[str, Any] | None = None

    def _resolve_api_key(self, config: object) -> str:
        del config
        return "test-key"

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del timeout
        self.last_url = url
        self.last_headers = headers
        self.last_payload = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": [["OpenAI platform"]]}'
                    }
                }
            ]
        }


class OptionalAuthRecordingProviderClient(OpenAICompatibleProviderClient):
    """Recording client that preserves provider-specific optional auth behavior."""

    def __init__(self) -> None:
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_payload: dict[str, Any] | None = None

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del timeout
        self.last_url = url
        self.last_headers = headers
        self.last_payload = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": [["OpenAI platform"]]}'
                    }
                }
            ]
        }


def test_openai_compatible_provider_sends_structured_output_request() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "openai"
    config.extraction.llm.model = "test-model"
    client = RecordingProviderClient()
    schema = {
        "type": "object",
        "properties": {"paragraphs": {"type": "array"}},
        "required": ["paragraphs"],
        "additionalProperties": False,
    }

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
        response_schema=schema,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'
    assert client.last_url == "https://api.openai.com/v1/chat/completions"
    assert client.last_headers is not None
    assert client.last_headers["Authorization"] == "Bearer test-key"
    assert client.last_payload is not None
    assert client.last_payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "labelgen_paragraph_concepts",
            "strict": True,
            "schema": schema,
        },
    }


def test_openai_compatible_provider_can_use_json_object_mode() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "openai"
    config.extraction.llm.model = "test-model"
    config.extraction.llm.output_contract_mode = "json_object"
    client = RecordingProviderClient()

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'
    assert client.last_payload is not None
    assert client.last_payload["response_format"] == {"type": "json_object"}


def test_openai_compatible_provider_can_use_prompt_only_mode() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "openai"
    config.extraction.llm.model = "test-model"
    config.extraction.llm.output_contract_mode = "prompt_only"
    client = RecordingProviderClient()

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'
    assert client.last_payload is not None
    assert "response_format" not in client.last_payload


class StructuredFallbackProviderClient(OpenAICompatibleProviderClient):
    """Provider client that rejects structured output once, then accepts prompt-only."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def _resolve_api_key(self, config: object) -> str:
        del config
        return "test-key"

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del url, headers, timeout
        self.payloads.append(payload)
        if "response_format" in payload:
            raise HTTPError(
                url="https://example.invalid/v1/chat/completions",
                code=400,
                msg="Unsupported response_format",
                hdrs=Message(),
                fp=None,
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": [["OpenAI platform"]]}'
                    }
                }
            ]
        }


def test_openai_compatible_provider_falls_back_when_json_schema_is_rejected() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "openai"
    config.extraction.llm.model = "test-model"
    client = StructuredFallbackProviderClient()
    schema = {
        "type": "object",
        "properties": {"paragraphs": {"type": "array"}},
        "required": ["paragraphs"],
        "additionalProperties": False,
    }

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
        response_schema=schema,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'
    assert len(client.payloads) == 3
    assert "response_format" in client.payloads[0]
    assert client.payloads[1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in client.payloads[2]


class JsonObjectFallbackProviderClient(OpenAICompatibleProviderClient):
    """Provider client that accepts json_object after json_schema is rejected."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def _resolve_api_key(self, config: object) -> str:
        del config
        return "test-key"

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del url, headers, timeout
        self.payloads.append(payload)
        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            response_format_dict = cast(dict[str, Any], response_format)
        else:
            response_format_dict = None
        if (
            response_format_dict is not None
            and response_format_dict.get("type") == "json_schema"
        ):
            raise HTTPError(
                url="https://example.invalid/v1/chat/completions",
                code=400,
                msg="Unsupported response_format",
                hdrs=Message(),
                fp=None,
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": [["OpenAI platform"]]}'
                    }
                }
            ]
        }


def test_openai_compatible_provider_stops_auto_fallback_at_json_object() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "ollama"
    config.extraction.llm.model = "qwen3.5:4b"
    client = JsonObjectFallbackProviderClient()
    schema = {
        "type": "object",
        "properties": {"paragraphs": {"type": "array"}},
        "required": ["paragraphs"],
        "additionalProperties": False,
    }

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
        response_schema=schema,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'
    assert len(client.payloads) == 2
    assert client.payloads[1]["response_format"] == {"type": "json_object"}


class EmptyContentFallbackProviderClient(OpenAICompatibleProviderClient):
    """Provider client that returns empty content until prompt-only mode is used."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def _resolve_api_key(self, config: object) -> str:
        del config
        return ""

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del url, headers, timeout
        self.payloads.append(payload)
        if "response_format" in payload:
            return {"choices": [{"message": {"content": ""}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": [["OpenAI platform"]]}'
                    }
                }
            ]
        }


def test_openai_compatible_provider_tries_weaker_contracts_after_empty_content() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "ollama"
    config.extraction.llm.model = "qwen3.5:4b"
    client = EmptyContentFallbackProviderClient()
    schema = {
        "type": "object",
        "properties": {"paragraphs": {"type": "array"}},
        "required": ["paragraphs"],
        "additionalProperties": False,
    }

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
        response_schema=schema,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'
    assert len(client.payloads) == 3
    assert client.payloads[1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in client.payloads[2]


class ReasoningOnlyProviderClient(OpenAICompatibleProviderClient):
    """Provider client that returns reasoning text when content is empty."""

    def _resolve_api_key(self, config: object) -> str:
        del config
        return ""

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del url, headers, payload, timeout
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": '{"paragraphs": [["OpenAI platform"]]}',
                    }
                }
            ]
        }


def test_openai_compatible_provider_can_fall_back_to_reasoning_text() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "ollama"
    config.extraction.llm.model = "qwen3.5:4b"
    client = ReasoningOnlyProviderClient()

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'


def test_openai_compatible_provider_raises_configuration_error_for_missing_api_key() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "openai"
    config.extraction.llm.model = "test-model"
    config.extraction.llm.api_key_env_var = "LABELGEN_TEST_MISSING_KEY"
    client = OpenAICompatibleProviderClient()

    try:
        client.complete_chat(
            messages=[{"role": "user", "content": "Extract concepts."}],
            config=config.extraction.llm,
        )
    except LLMProviderConfigurationError as error:
        assert error.provider == "openai"
        assert "LABELGEN_TEST_MISSING_KEY" in str(error)
    else:
        raise AssertionError("Expected a configuration error for a missing API key.")


class AlwaysHTTPErrorProviderClient(OpenAICompatibleProviderClient):
    """Provider client that always returns a fixed HTTP error."""

    def _resolve_api_key(self, config: object) -> str:
        del config
        return "test-key"

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del url, headers, payload, timeout
        raise HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=Message(),
            fp=None,
        )


def test_openai_compatible_provider_preserves_http_status_diagnostics() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "mistral"
    config.extraction.llm.model = "test-model"
    config.extraction.llm.max_retries = 0
    client = AlwaysHTTPErrorProviderClient()

    try:
        client.complete_chat(
            messages=[{"role": "user", "content": "Extract concepts."}],
            config=config.extraction.llm,
        )
    except LLMProviderRetryExhaustedError as error:
        assert error.provider == "mistral"
        assert isinstance(error.last_error, LLMProviderHTTPStatusError)
        assert error.last_error.status_code == 429
        assert error.last_error.response_summary is None
        assert "mistral" in str(error)
    else:
        raise AssertionError("Expected retry exhaustion with an HTTP status error.")


class AlwaysURLErrorProviderClient(OpenAICompatibleProviderClient):
    """Provider client that always raises a transport error."""

    def _resolve_api_key(self, config: object) -> str:
        del config
        return "test-key"

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del url, headers, payload, timeout
        raise URLError("connection reset by peer")


def test_openai_compatible_provider_preserves_transport_diagnostics() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "qwen"
    config.extraction.llm.model = "test-model"
    config.extraction.llm.max_retries = 0
    client = AlwaysURLErrorProviderClient()

    try:
        client.complete_chat(
            messages=[{"role": "user", "content": "Extract concepts."}],
            config=config.extraction.llm,
        )
    except LLMProviderRetryExhaustedError as error:
        assert error.provider == "qwen"
        assert isinstance(error.last_error, LLMProviderTransportError)
        assert "connection reset by peer" in str(error.last_error)
        assert "qwen" in str(error)
    else:
        raise AssertionError("Expected retry exhaustion with a transport error.")


def test_openai_compatible_provider_uses_default_ollama_base_url() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "ollama"
    config.extraction.llm.model = "llama3.1"
    client = RecordingProviderClient()

    content = client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
    )

    assert content == '{"paragraphs": [["OpenAI platform"]]}'
    assert client.last_url == "http://localhost:11434/v1/chat/completions"


def test_openai_compatible_provider_does_not_require_ollama_api_key() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.provider = "ollama"
    config.extraction.llm.model = "llama3.1"
    config.extraction.llm.api_key_env_var = "LABELGEN_UNUSED_OLLAMA_KEY"
    client = OptionalAuthRecordingProviderClient()

    client.complete_chat(
        messages=[{"role": "user", "content": "Extract concepts."}],
        config=config.extraction.llm,
    )

    assert client.last_headers is not None
    assert "Authorization" not in client.last_headers
