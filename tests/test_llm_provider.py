"""Tests for the OpenAI-compatible LLM provider client."""

from __future__ import annotations

from typing import Any

from labelgen import LabelGeneratorConfig
from labelgen.extraction.llm_provider import OpenAICompatibleProviderClient


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
