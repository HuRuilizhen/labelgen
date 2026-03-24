"""Provider-backed concept extraction using LLMs."""

from __future__ import annotations

import hashlib
import json
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
        for batch in self._iter_batches(paragraphs, self._config.llm.batch_size):
            concept_lists = self._extract_batch_concepts(batch)
            for paragraph, concepts in zip(batch, concept_lists, strict=True):
                mentions.extend(self._build_mentions(paragraph, concepts))
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

    def _extract_batch_concepts(self, paragraphs: list[Paragraph]) -> list[list[str]]:
        """Extract concept strings for one paragraph batch."""

        cache_path = self._cache_path(paragraphs)
        if cache_path is not None and cache_path.exists():
            return self._load_cached_batch(cache_path, len(paragraphs))

        content = self._client.complete_chat(
            messages=self._build_messages(paragraphs),
            config=self._config.llm,
        )
        concept_lists = self._parse_provider_output(content, len(paragraphs))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"paragraphs": concept_lists}, indent=2),
                encoding="utf-8",
            )
        return concept_lists

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

        concept_lists: list[list[str]] = [[] for _ in range(paragraph_count)]
        for item in cast(list[object], paragraphs):
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, Any], item)
            paragraph_index = item_dict.get("paragraph_index")
            concepts = item_dict.get("concepts")
            if not isinstance(paragraph_index, int):
                continue
            if paragraph_index < 0 or paragraph_index >= paragraph_count:
                continue
            if not isinstance(concepts, list):
                continue

            seen: set[str] = set()
            parsed_concepts: list[str] = []
            for concept in cast(list[object], concepts):
                if not isinstance(concept, str):
                    continue
                normalized = normalize_surface(concept, lowercase=self._config.lowercase)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                parsed_concepts.append(concept.strip())
                if len(parsed_concepts) >= self._config.llm.max_concepts_per_paragraph:
                    break
            concept_lists[paragraph_index] = parsed_concepts
        return concept_lists

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

    def _cache_path(self, paragraphs: list[Paragraph]) -> Path | None:
        """Return the cache path for one request batch when caching is enabled."""

        if not self._config.llm.cache_enabled or self._config.llm.cache_dir is None:
            return None
        key_payload = {
            "provider": self._config.llm.provider,
            "model": self._config.llm.model,
            "prompt_version": self._config.llm.prompt_version,
            "temperature": self._config.llm.temperature,
            "max_output_tokens": self._config.llm.max_output_tokens,
            "paragraphs": [paragraph.text for paragraph in paragraphs],
        }
        digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        return Path(self._config.llm.cache_dir) / f"{digest}.json"

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
