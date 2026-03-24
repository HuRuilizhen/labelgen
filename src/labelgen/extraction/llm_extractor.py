"""Provider-backed concept extraction using LLMs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from labelgen.config import ExtractionConfig, LLMExtractionConfig
from labelgen.extraction.concept_extractor import ConceptExtractor
from labelgen.extraction.llm_provider import LLMProviderClient, build_provider_client
from labelgen.extraction.normalization import normalize_surface
from labelgen.types import ConceptMention, Paragraph

_DEFAULT_PROMPT_TEMPLATE = "\n".join(
    [
        "Return a JSON object with this exact schema:",
        '{{"paragraphs": [["concept 1", "concept 2"], ["concept 3"]]}}',
        "Rules:",
        "- Extract concise technical concepts suitable for paragraph labeling.",
        "- Prefer products, components, errors, operations, entities, and domain nouns.",
        "- Exclude URLs, generic support boilerplate, pronouns, and section headers.",
        "- Preserve the original concept wording when useful.",
        "- Return at most {max_concepts_per_paragraph} concepts per paragraph.",
        "",
        "{paragraphs_block}",
    ]
)


@dataclass(slots=True)
class _BatchExtractionPayload:
    """Structured intermediate data for one LLM extraction batch."""

    source: str
    cache_digest: str | None
    messages: list[dict[str, str]]
    raw_response_text: str | None
    concept_lists: list[list[str]]


class LLMConceptExtractor(ConceptExtractor):
    """Concept extractor backed by a provider LLM."""

    def __init__(
        self,
        config: ExtractionConfig,
        *,
        client: LLMProviderClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or build_provider_client(config.llm)

    def extract(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract concepts from paragraphs through the configured provider."""

        self._validate_config(self._config.llm)
        mentions: list[ConceptMention] = []
        for batch_index, batch in enumerate(
            self._iter_batches(paragraphs, self._config.llm.batch_size)
        ):
            payload = self._extract_batch_concepts(batch, batch_index=batch_index)
            batch_mentions: list[ConceptMention] = []
            for paragraph, concepts in zip(batch, payload.concept_lists, strict=True):
                batch_mentions.extend(self._build_mentions(paragraph, concepts))
            self._write_artifact(
                batch_index=batch_index,
                paragraphs=batch,
                payload=payload,
                mentions=batch_mentions,
            )
            mentions.extend(batch_mentions)
        return mentions

    def _validate_config(self, config: LLMExtractionConfig) -> None:
        """Validate required LLM extraction settings."""

        if not config.model:
            raise RuntimeError("LLM extraction requires a configured model name.")
        if config.batch_size <= 0:
            raise RuntimeError("LLM extraction batch_size must be positive.")
        if config.max_concepts_per_paragraph <= 0:
            raise RuntimeError("LLM extraction max_concepts_per_paragraph must be positive.")

    def _iter_batches(
        self,
        paragraphs: list[Paragraph],
        batch_size: int,
    ) -> list[list[Paragraph]]:
        """Split paragraphs into provider request batches."""

        return [
            paragraphs[index : index + batch_size]
            for index in range(0, len(paragraphs), batch_size)
        ]

    def _extract_batch_concepts(
        self,
        paragraphs: list[Paragraph],
        *,
        batch_index: int,
    ) -> _BatchExtractionPayload:
        """Extract concept strings for one paragraph batch."""

        messages = self._build_messages(paragraphs)
        cache_digest = self._cache_digest(paragraphs)
        cache_path = self._cache_path(cache_digest)
        if cache_path is not None and cache_path.exists():
            return _BatchExtractionPayload(
                source="cache",
                cache_digest=cache_digest,
                messages=messages,
                raw_response_text=None,
                concept_lists=self._load_cached_batch(cache_path, len(paragraphs)),
            )

        content = self._client.complete_chat(
            messages=messages,
            config=self._config.llm,
        )
        try:
            concept_lists = self._parse_provider_output(content, len(paragraphs))
        except RuntimeError as error:
            self._write_failure_artifact(
                batch_index=batch_index,
                paragraphs=paragraphs,
                cache_digest=cache_digest,
                messages=messages,
                raw_response_text=content,
                error_message=str(error),
            )
            raise
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"paragraphs": concept_lists}, indent=2),
                encoding="utf-8",
            )
        return _BatchExtractionPayload(
            source="provider",
            cache_digest=cache_digest,
            messages=messages,
            raw_response_text=content,
            concept_lists=concept_lists,
        )

    def _build_messages(self, paragraphs: list[Paragraph]) -> list[dict[str, str]]:
        """Build the system and user messages for one extraction batch."""

        system_message = (
            "You extract salient technical concepts from paragraphs for downstream "
            "topic labeling. Return JSON only."
        )
        prompt_template = self._config.llm.prompt_template or _DEFAULT_PROMPT_TEMPLATE
        paragraph_lines = [
            f"Paragraph {index}: {paragraph.text}"
            for index, paragraph in enumerate(paragraphs)
        ]
        user_message = prompt_template.format(
            max_concepts_per_paragraph=self._config.llm.max_concepts_per_paragraph,
            paragraphs_block="\n".join(paragraph_lines),
        )
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _parse_provider_output(self, content: str, paragraph_count: int) -> list[list[str]]:
        """Parse a provider response into per-paragraph concept text lists."""

        data = self._load_json_object(content)
        paragraphs = data.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise RuntimeError("LLM extraction output must contain a paragraphs list.")

        raw_lists = cast(list[object], paragraphs)
        if len(raw_lists) != paragraph_count:
            raise RuntimeError("LLM extraction output must align with the input batch size.")

        concept_lists: list[list[str]] = []
        for item in raw_lists:
            if not isinstance(item, list):
                raise RuntimeError("Each LLM extraction paragraph entry must be a list of strings.")
            concept_lists.append(self._parse_concept_list(cast(list[object], item)))
        return concept_lists

    def _parse_concept_list(self, values: list[object]) -> list[str]:
        """Parse one ordered concept list from a provider response."""

        seen: set[str] = set()
        parsed_concepts: list[str] = []
        for concept in values:
            if not isinstance(concept, str):
                continue
            normalized = normalize_surface(concept, lowercase=self._config.lowercase)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            parsed_concepts.append(concept.strip())
            if len(parsed_concepts) >= self._config.llm.max_concepts_per_paragraph:
                break
        return parsed_concepts

    def _load_json_object(self, content: str) -> dict[str, Any]:
        """Parse JSON or extract a JSON object from a textual provider response."""

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or start >= end:
                raise RuntimeError("LLM extraction response did not contain valid JSON.") from None
            data = json.loads(content[start : end + 1])
        if not isinstance(data, dict):
            raise RuntimeError("LLM extraction response must decode to a JSON object.")
        return cast(dict[str, Any], data)

    def _build_mentions(self, paragraph: Paragraph, concepts: list[str]) -> list[ConceptMention]:
        """Convert parsed concept text into mention models."""

        mentions: list[ConceptMention] = []
        for concept in concepts:
            normalized = normalize_surface(concept, lowercase=self._config.lowercase)
            mention = ConceptMention(
                paragraph_id=paragraph.id,
                concept_id=self._make_concept_id(normalized),
                surface=concept,
                normalized=normalized,
                kind="llm_concept",
            )
            mentions.append(mention)
        return mentions

    def _make_concept_id(self, normalized: str) -> str:
        """Create a stable concept identifier for an LLM concept."""

        digest = hashlib.sha256(f"llm_concept:{normalized}".encode()).hexdigest()
        return digest[:16]

    def _cache_digest(self, paragraphs: list[Paragraph]) -> str | None:
        """Return the cache digest for one request batch when caching is enabled."""

        if not self._config.llm.cache_enabled:
            return None
        key_payload = {
            "provider": self._config.llm.provider,
            "model": self._config.llm.model,
            "prompt_version": self._config.llm.prompt_version,
            "prompt_template": self._config.llm.prompt_template,
            "temperature": self._config.llm.temperature,
            "max_output_tokens": self._config.llm.max_output_tokens,
            "max_concepts_per_paragraph": self._config.llm.max_concepts_per_paragraph,
            "paragraphs": [paragraph.text for paragraph in paragraphs],
        }
        return hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()

    def _cache_path(self, cache_digest: str | None) -> Path | None:
        """Return the cache path for one request batch when caching is enabled."""

        if cache_digest is None or self._config.llm.cache_dir is None:
            return None
        return Path(self._config.llm.cache_dir) / f"{cache_digest}.json"

    def _load_cached_batch(self, path: Path, paragraph_count: int) -> list[list[str]]:
        """Load cached per-paragraph concepts from disk."""

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("Cached LLM extraction payload must be a JSON object.")
        cached_payload = cast(dict[str, Any], data)
        paragraphs_value = cached_payload.get("paragraphs")
        if not isinstance(paragraphs_value, list):
            raise RuntimeError("Cached LLM extraction payload has an unexpected shape.")
        cached_paragraphs = cast(list[object], paragraphs_value)
        if len(cached_paragraphs) != paragraph_count:
            raise RuntimeError("Cached LLM extraction payload has an unexpected shape.")
        loaded: list[list[str]] = []
        for item in cached_paragraphs:
            if not isinstance(item, list):
                raise RuntimeError("Cached LLM extraction paragraph concepts must be a list.")
            concepts = [concept for concept in cast(list[object], item) if isinstance(concept, str)]
            loaded.append(concepts)
        return loaded

    def _write_artifact(
        self,
        *,
        batch_index: int,
        paragraphs: list[Paragraph],
        payload: _BatchExtractionPayload,
        mentions: list[ConceptMention],
    ) -> None:
        """Write a structured extraction artifact when recording is enabled."""

        artifact_dir = self._artifact_dir()
        if artifact_dir is None:
            return
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        cache_suffix = payload.cache_digest[:12] if payload.cache_digest is not None else "nocache"
        artifact_path = artifact_dir / f"{timestamp}-batch{batch_index:03d}-{cache_suffix}.json"
        artifact_payload = {
            "artifact_type": "llm_extraction_batch",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": payload.source,
            "cache_digest": payload.cache_digest,
            "provider": self._config.llm.provider,
            "model": self._config.llm.model,
            "prompt_version": self._config.llm.prompt_version,
            "prompt_template": self._config.llm.prompt_template,
            "messages": payload.messages,
            "paragraphs": [asdict(paragraph) for paragraph in paragraphs],
            "raw_response_text": payload.raw_response_text,
            "parsed_concepts": payload.concept_lists,
            "mentions": [asdict(mention) for mention in mentions],
        }
        artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    def _artifact_dir(self) -> Path | None:
        """Resolve the optional artifact output directory."""

        if not self._config.llm.record_extraction_artifacts:
            return None
        if self._config.llm.artifact_dir is None:
            return None
        return Path(self._config.llm.artifact_dir)

    def _write_failure_artifact(
        self,
        *,
        batch_index: int,
        paragraphs: list[Paragraph],
        cache_digest: str | None,
        messages: list[dict[str, str]],
        raw_response_text: str,
        error_message: str,
    ) -> None:
        """Write a structured failure artifact when parsing fails."""

        artifact_dir = self._artifact_dir()
        if artifact_dir is None:
            return
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        cache_suffix = cache_digest[:12] if cache_digest is not None else "nocache"
        artifact_path = (
            artifact_dir / f"{timestamp}-batch{batch_index:03d}-{cache_suffix}-error.json"
        )
        artifact_payload = {
            "artifact_type": "llm_extraction_batch_error",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "provider": self._config.llm.provider,
            "model": self._config.llm.model,
            "prompt_version": self._config.llm.prompt_version,
            "prompt_template": self._config.llm.prompt_template,
            "cache_digest": cache_digest,
            "messages": messages,
            "paragraphs": [asdict(paragraph) for paragraph in paragraphs],
            "raw_response_text": raw_response_text,
            "error_message": error_message,
        }
        artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
