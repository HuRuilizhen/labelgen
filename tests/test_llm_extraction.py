"""Tests for LLM-backed concept extraction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from labelgen import LabelGenerator, LabelGeneratorConfig
from labelgen.extraction.llm_extractor import LLMConceptExtractor
from labelgen.extraction.llm_provider import LLMProviderClient
from labelgen.types import Paragraph


class FakeLLMProviderClient(LLMProviderClient):
    """Simple fake provider client returning predetermined JSON."""

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.call_count = 0
        self.messages: list[dict[str, str]] = []
        self.response_schema: dict[str, object] | None = None

    def complete_chat(
        self,
        *,
        messages: list[dict[str, str]],
        config: object,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        self.messages = messages
        del config
        self.response_schema = response_schema
        self.call_count += 1
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


def test_llm_extractor_returns_llm_concept_mentions(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient(
        {
            "paragraphs": [["OpenAI platform", "developer tooling"]]
        }
    )
    extractor = LLMConceptExtractor(config.extraction, client=client)

    mentions = extractor.extract([Paragraph(id="p1", text="OpenAI builds developer tooling.")])

    assert client.call_count == 1
    assert [mention.normalized for mention in mentions] == [
        "openai platform",
        "developer tooling",
    ]
    assert all(mention.kind == "llm_concept" for mention in mentions)


def test_llm_extractor_uses_custom_prompt_template(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    config.extraction.llm.prompt_template = (
        "Extract concepts.\n"
        "Cap: {max_concepts_per_paragraph}\n"
        "{paragraphs_block}"
    )
    client = FakeLLMProviderClient({"paragraphs": [["OpenAI platform"]]})
    extractor = LLMConceptExtractor(config.extraction, client=client)

    extractor.extract([Paragraph(id="p1", text="OpenAI builds developer tooling.")])

    assert len(client.messages) == 2
    assert "Cap: 12" in client.messages[1]["content"]
    assert "Paragraph 0: OpenAI builds developer tooling." in client.messages[1]["content"]


def test_llm_extractor_default_prompt_includes_exact_paragraph_count(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient({"paragraphs": [["OpenAI platform"]]})
    extractor = LLMConceptExtractor(config.extraction, client=client)

    extractor.extract([Paragraph(id="p1", text="OpenAI builds developer tooling.")])

    prompt = client.messages[1]["content"]
    assert '"paragraphs" must contain exactly 1 arrays.' in prompt
    assert '{"paragraphs": [["concept 1", "concept 2"]]}' in prompt
    assert '{"paragraphs": [[]]}' in prompt
    assert "Do not return extra arrays" in prompt


def test_llm_extractor_passes_structured_schema_to_provider(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_enabled = False
    config.extraction.llm.max_concepts_per_paragraph = 3
    client = FakeLLMProviderClient({"paragraphs": [["OpenAI platform"]]})
    extractor = LLMConceptExtractor(config.extraction, client=client)

    extractor.extract([Paragraph(id="p1", text="OpenAI builds developer tooling.")])

    assert client.response_schema is not None
    assert client.response_schema["type"] == "object"
    properties = cast(dict[str, Any], client.response_schema["properties"])
    paragraphs_schema = cast(dict[str, Any], properties["paragraphs"])
    assert paragraphs_schema["minItems"] == 1
    assert paragraphs_schema["maxItems"] == 1
    items_schema = cast(dict[str, Any], paragraphs_schema["items"])
    assert items_schema["maxItems"] == 3


def test_llm_extractor_uses_disk_cache(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path / "cache")
    client = FakeLLMProviderClient(
        {
            "paragraphs": [
                ["OpenAI platform"],
            ]
        }
    )
    extractor = LLMConceptExtractor(config.extraction, client=client)
    paragraphs = [Paragraph(id="p1", text="OpenAI builds platforms.")]

    first = extractor.extract(paragraphs)
    second = extractor.extract(paragraphs)

    assert [mention.normalized for mention in first] == [mention.normalized for mention in second]
    assert client.call_count == 1


def test_llm_extractor_accepts_empty_paragraphs_array_for_single_input(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient({"paragraphs": []})
    extractor = LLMConceptExtractor(config.extraction, client=client)

    mentions = extractor.extract([Paragraph(id="p1", text="Boilerplate paragraph.")])

    assert mentions == []


def test_llm_extractor_accepts_json_with_trailing_garbage(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient('{"paragraphs": [[]]}}')
    extractor = LLMConceptExtractor(config.extraction, client=client)

    mentions = extractor.extract([Paragraph(id="p1", text="Boilerplate paragraph.")])

    assert mentions == []


def test_llm_extractor_accepts_markdown_fenced_json(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient('```json\n{"paragraphs": [[]]}\n```')
    extractor = LLMConceptExtractor(config.extraction, client=client)

    mentions = extractor.extract([Paragraph(id="p1", text="Boilerplate paragraph.")])

    assert mentions == []


def test_llm_extractor_prefers_last_valid_paragraphs_object_in_free_text(
    tmp_path: Path,
) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient(
        "Reasoning:\n"
        'placeholder {"paragraphs": [["concept 1"]]}\n'
        'final {"paragraphs": [["OpenAI platform"]]}'
    )
    extractor = LLMConceptExtractor(config.extraction, client=client)

    mentions = extractor.extract([Paragraph(id="p1", text="OpenAI builds APIs for developers.")])

    assert [mention.normalized for mention in mentions] == ["openai platform"]


def test_llm_extractor_recovers_missing_closing_delimiters(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient('{"paragraphs": [[]]')
    extractor = LLMConceptExtractor(config.extraction, client=client)

    mentions = extractor.extract([Paragraph(id="p1", text="Boilerplate paragraph.")])

    assert mentions == []


def test_llm_extractor_can_record_structured_artifacts(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_enabled = False
    config.extraction.llm.record_extraction_artifacts = True
    config.extraction.llm.record_raw_response_text = True
    config.extraction.llm.record_paragraph_text = True
    config.extraction.llm.record_paragraph_metadata = True
    config.extraction.llm.artifact_dir = str(tmp_path / "artifacts")
    client = FakeLLMProviderClient({"paragraphs": [["OpenAI platform", "developer tooling"]]})
    extractor = LLMConceptExtractor(config.extraction, client=client)

    mentions = extractor.extract([Paragraph(id="p1", text="OpenAI builds developer tooling.")])

    artifact_files = sorted((tmp_path / "artifacts").glob("*.json"))
    assert len(artifact_files) == 1
    artifact = json.loads(artifact_files[0].read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "llm_extraction_batch"
    assert artifact["source"] == "provider"
    assert artifact["provider"] == "openai"
    assert artifact["model"] == "test-model"
    assert artifact["paragraphs"][0]["id"] == "p1"
    assert artifact["raw_response_text"] == json.dumps(
        {"paragraphs": [["OpenAI platform", "developer tooling"]]}
    )
    assert artifact["parsed_concepts"] == [["OpenAI platform", "developer tooling"]]
    assert [item["normalized"] for item in artifact["mentions"]] == [
        mention.normalized for mention in mentions
    ]


def test_llm_extractor_artifacts_are_json_safe(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_enabled = False
    config.extraction.llm.record_extraction_artifacts = True
    config.extraction.llm.record_paragraph_metadata = True
    config.extraction.llm.artifact_dir = str(tmp_path / "artifacts")
    client = FakeLLMProviderClient({"paragraphs": [["OpenAI platform"]]})
    extractor = LLMConceptExtractor(config.extraction, client=client)
    paragraphs = [
        Paragraph(
            id="p1",
            text="OpenAI builds developer tooling.",
            metadata={
                "seen_at": datetime(2026, 3, 25, tzinfo=UTC),
                "tags": {"alpha", "beta"},
            },
        )
    ]

    extractor.extract(paragraphs)

    artifact_files = sorted((tmp_path / "artifacts").glob("*.json"))
    artifact = json.loads(artifact_files[0].read_text(encoding="utf-8"))
    assert artifact["paragraphs"][0]["metadata"]["seen_at"] == "2026-03-25T00:00:00+00:00"
    assert artifact["paragraphs"][0]["metadata"]["tags"] == ["alpha", "beta"]


def test_llm_extractor_records_failure_artifacts(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_enabled = False
    config.extraction.llm.record_extraction_artifacts = True
    config.extraction.llm.record_raw_response_text = True
    config.extraction.llm.record_paragraph_text = True
    config.extraction.llm.artifact_dir = str(tmp_path / "artifacts")
    client = FakeLLMProviderClient({"paragraphs": [["OpenAI"], ["developer tooling"]]})
    extractor = LLMConceptExtractor(config.extraction, client=client)

    with pytest.raises(RuntimeError, match="align with the input batch size"):
        extractor.extract([Paragraph(id="p1", text="OpenAI builds developer tooling.")])

    artifact_files = sorted((tmp_path / "artifacts").glob("*-error.json"))
    assert len(artifact_files) == 1
    artifact = json.loads(artifact_files[0].read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "llm_extraction_batch_error"
    assert artifact["error_message"] == (
        "RuntimeError: LLM extraction output must align with the input batch size."
    )
    assert artifact["raw_response_text"] == json.dumps(
        {"paragraphs": [["OpenAI"], ["developer tooling"]]}
    )
    assert artifact["paragraphs"][0]["id"] == "p1"


def test_llm_extractor_artifact_safety_controls_suppress_sensitive_fields(
    tmp_path: Path,
) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_enabled = False
    config.extraction.llm.record_extraction_artifacts = True
    config.extraction.llm.artifact_dir = str(tmp_path / "artifacts")
    client = FakeLLMProviderClient({"paragraphs": [["OpenAI platform"]]})
    extractor = LLMConceptExtractor(config.extraction, client=client)

    extractor.extract(
        [
            Paragraph(
                id="p1",
                text="OpenAI builds developer tooling.",
                metadata={"source": "internal"},
            )
        ]
    )

    artifact_files = sorted((tmp_path / "artifacts").glob("*.json"))
    artifact = json.loads(artifact_files[0].read_text(encoding="utf-8"))
    assert artifact["raw_response_text"] is None
    assert artifact["paragraphs"] == [{"id": "p1"}]
    assert artifact["messages"][1]["content"] == "[omitted because record_paragraph_text=False]"


def test_llm_cache_key_includes_max_concepts_per_paragraph(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path / "cache")
    config.extraction.llm.max_concepts_per_paragraph = 1
    first_client = FakeLLMProviderClient({"paragraphs": [["OpenAI", "developer tooling"]]})
    extractor = LLMConceptExtractor(config.extraction, client=first_client)
    paragraphs = [Paragraph(id="p1", text="OpenAI builds developer tooling.")]

    first_mentions = extractor.extract(paragraphs)

    config.extraction.llm.max_concepts_per_paragraph = 2
    second_client = FakeLLMProviderClient({"paragraphs": [["OpenAI", "developer tooling"]]})
    second_extractor = LLMConceptExtractor(config.extraction, client=second_client)
    second_mentions = second_extractor.extract(paragraphs)

    assert [mention.normalized for mention in first_mentions] == ["openai"]
    assert [mention.normalized for mention in second_mentions] == [
        "openai",
        "developer tooling",
    ]
    assert first_client.call_count == 1
    assert second_client.call_count == 1


def test_llm_cache_key_includes_effective_default_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path / "cache")
    paragraphs = [Paragraph(id="p1", text="OpenAI builds developer tooling.")]

    first_client = FakeLLMProviderClient({"paragraphs": [["OpenAI"]]})
    extractor = LLMConceptExtractor(config.extraction, client=first_client)
    first_mentions = extractor.extract(paragraphs)

    monkeypatch.setattr(
        "labelgen.extraction.llm_extractor._DEFAULT_PROMPT_TEMPLATE",
        "Different built-in prompt.\n{paragraphs_block}",
    )
    second_client = FakeLLMProviderClient({"paragraphs": [["OpenAI", "developer tooling"]]})
    second_extractor = LLMConceptExtractor(config.extraction, client=second_client)
    second_mentions = second_extractor.extract(paragraphs)

    assert [mention.normalized for mention in first_mentions] == ["openai"]
    assert [mention.normalized for mention in second_mentions] == [
        "openai",
        "developer tooling",
    ]
    assert first_client.call_count == 1
    assert second_client.call_count == 1


def test_llm_cache_key_includes_output_contract_mode(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path / "cache")
    paragraphs = [Paragraph(id="p1", text="OpenAI builds developer tooling.")]

    first_client = FakeLLMProviderClient({"paragraphs": [["OpenAI"]]})
    extractor = LLMConceptExtractor(config.extraction, client=first_client)
    first_mentions = extractor.extract(paragraphs)

    config.extraction.llm.output_contract_mode = "prompt_only"
    second_client = FakeLLMProviderClient({"paragraphs": [["OpenAI", "developer tooling"]]})
    second_extractor = LLMConceptExtractor(config.extraction, client=second_client)
    second_mentions = second_extractor.extract(paragraphs)

    assert [mention.normalized for mention in first_mentions] == ["openai"]
    assert [mention.normalized for mention in second_mentions] == [
        "openai",
        "developer tooling",
    ]
    assert first_client.call_count == 1
    assert second_client.call_count == 1


def test_label_generator_uses_llm_extractor_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeLLMProviderClient(
        {
            "paragraphs": [
                ["OpenAI", "language models"],
                ["language models", "production systems"],
            ]
        }
    )
    def _build_fake_client(config: object) -> LLMProviderClient:
        del config
        return client

    monkeypatch.setattr(
        "labelgen.extraction.llm_extractor.build_provider_client",
        _build_fake_client,
    )

    config = LabelGeneratorConfig(
        extractor_mode="llm",
        use_graph_community_detection=False,
    )
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_enabled = False
    config.extraction.llm.cache_dir = str(tmp_path / "cache")

    generator = LabelGenerator(config)
    result = generator.fit_transform(
        [
            "OpenAI builds language models.",
            "Production systems use language models.",
        ]
    )

    assert generator.extractor_name == "LLMConceptExtractor"
    assert result.concepts
    assert result.paragraph_labels[0].label_ids


def test_llm_mode_requires_configured_model() -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = ""
    config.use_graph_community_detection = False
    generator = LabelGenerator(config)

    with pytest.raises(RuntimeError, match="requires a configured model"):
        generator.fit_transform(["OpenAI builds language models."])


def test_llm_extractor_requires_positional_paragraph_lists(tmp_path: Path) -> None:
    config = LabelGeneratorConfig(extractor_mode="llm")
    config.extraction.llm.model = "test-model"
    config.extraction.llm.cache_dir = str(tmp_path)
    client = FakeLLMProviderClient(
        {
            "paragraphs": [
                {"paragraph_index": 0, "concepts": ["OpenAI platform"]},
            ]
        }
    )
    extractor = LLMConceptExtractor(config.extraction, client=client)

    with pytest.raises(RuntimeError, match="must be a list of strings"):
        extractor.extract([Paragraph(id="p1", text="OpenAI builds platforms.")])
