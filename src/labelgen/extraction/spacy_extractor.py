"""spaCy-backed concept extraction with deterministic fallback rules."""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from labelgen.config import ExtractionConfig
from labelgen.extraction.concept_extractor import ConceptExtractor
from labelgen.extraction.normalization import normalize_surface
from labelgen.types import ConceptMention, Paragraph

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class _Token:
    """A regex-tokenized text unit with offsets."""

    text: str
    start: int
    end: int


class SpacyConceptExtractor(ConceptExtractor):
    """Concept extractor implementation backed by spaCy when available."""

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._nlp: object | None = self._load_spacy_pipeline() if config.prefer_spacy else None

    def extract(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract mentions from paragraphs."""

        if self._nlp is not None:
            return self._extract_with_spacy(paragraphs)
        return self._extract_with_rules(paragraphs)

    def _load_spacy_pipeline(self) -> object | None:
        """Load a spaCy pipeline if the dependency and model are available."""

        try:
            spacy = importlib.import_module("spacy")
        except ImportError:
            return None

        try:
            load = getattr(spacy, "load", None)
            if not callable(load):
                return None
            return load(self._config.spacy_model_name)
        except OSError:
            return None

    def _extract_with_spacy(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract mentions with spaCy entities and noun chunks."""

        if self._nlp is None:
            return []

        mentions: list[ConceptMention] = []
        pipe = getattr(self._nlp, "pipe", None)
        if not callable(pipe):
            return self._extract_with_rules(paragraphs)

        docs = self._iter_objects(pipe(paragraph.text for paragraph in paragraphs))
        for paragraph, doc in zip(paragraphs, docs, strict=True):
            mentions.extend(self._extract_doc_mentions(paragraph.id, doc))
        return mentions

    def _extract_doc_mentions(self, paragraph_id: str, doc: object) -> list[ConceptMention]:
        """Extract spaCy spans as mentions while preserving stable ordering."""

        mentions: list[ConceptMention] = []
        seen: set[tuple[int, int, str]] = set()

        for span in self._iter_objects(getattr(doc, "ents", ())):
            mention = self._make_mention(
                paragraph_id,
                str(getattr(span, "text", "")),
                "entity",
                self._as_int(getattr(span, "start_char", None)),
                self._as_int(getattr(span, "end_char", None)),
            )
            key = (mention.start or 0, mention.end or 0, mention.kind)
            if key not in seen:
                seen.add(key)
                mentions.append(mention)

        noun_chunks = self._iter_noun_chunks(doc)
        for chunk in noun_chunks:
            mention = self._make_mention(
                paragraph_id,
                str(getattr(chunk, "text", "")),
                "noun_phrase",
                self._as_int(getattr(chunk, "start_char", None)),
                self._as_int(getattr(chunk, "end_char", None)),
            )
            key = (mention.start or 0, mention.end or 0, mention.kind)
            if key not in seen:
                seen.add(key)
                mentions.append(mention)

        return sorted(mentions, key=lambda item: (item.start or -1, item.end or -1, item.kind))

    def _iter_noun_chunks(self, doc: object) -> tuple[object, ...]:
        """Return noun chunks if the loaded pipeline supports them."""

        noun_chunks_attribute = "noun_chunks"
        try:
            noun_chunks = getattr(doc, noun_chunks_attribute)
        except (AttributeError, ValueError):
            return ()
        return self._iter_objects(noun_chunks)

    def _extract_with_rules(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract mentions using deterministic regex heuristics."""

        mentions: list[ConceptMention] = []
        for paragraph in paragraphs:
            tokens = [self._token_from_match(match) for match in _TOKEN_RE.finditer(paragraph.text)]
            mentions.extend(self._extract_rule_mentions(paragraph.id, tokens))
        return mentions

    def _extract_rule_mentions(
        self,
        paragraph_id: str,
        tokens: list[_Token],
    ) -> list[ConceptMention]:
        """Extract entity and noun phrase mentions from tokenized text."""

        mentions: list[ConceptMention] = []
        seen: set[tuple[int, int, str]] = set()

        for start_index, end_index in self._iter_entity_spans(tokens):
            mention = self._mention_from_tokens(
                paragraph_id,
                tokens[start_index:end_index],
                "entity",
            )
            key = (mention.start or 0, mention.end or 0, mention.kind)
            if key not in seen:
                seen.add(key)
                mentions.append(mention)

        for start_index, end_index in self._iter_candidate_phrase_spans(tokens):
            mention = self._mention_from_tokens(
                paragraph_id,
                tokens[start_index:end_index],
                "noun_phrase",
            )
            key = (mention.start or 0, mention.end or 0, mention.kind)
            if key not in seen:
                seen.add(key)
                mentions.append(mention)

        return sorted(mentions, key=lambda item: (item.start or -1, item.end or -1, item.kind))

    def _iter_entity_spans(self, tokens: list[_Token]) -> list[tuple[int, int]]:
        """Yield capitalized token spans as a lightweight entity heuristic."""

        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            normalized = normalize_surface(token.text, lowercase=True)
            if normalized in _STOPWORDS or not self._is_entity_like(token.text):
                index += 1
                continue

            end_index = index + 1
            while end_index < len(tokens):
                candidate = tokens[end_index]
                candidate_normalized = normalize_surface(candidate.text, lowercase=True)
                if candidate_normalized in _STOPWORDS or not self._is_entity_like(candidate.text):
                    break
                if end_index - index >= self._config.max_phrase_length:
                    break
                end_index += 1

            spans.append((index, end_index))
            index = end_index

        return spans

    def _iter_candidate_phrase_spans(self, tokens: list[_Token]) -> list[tuple[int, int]]:
        """Yield content-word spans as a noun-phrase heuristic."""

        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(tokens):
            if self._is_stopword(tokens[index].text):
                index += 1
                continue

            end_index = index
            while end_index < len(tokens) and not self._is_stopword(tokens[end_index].text):
                end_index += 1

            span_length = end_index - index
            if span_length <= self._config.max_phrase_length:
                spans.append((index, end_index))
            else:
                window_size = self._config.max_phrase_length
                for window_start in range(index, end_index - window_size + 1):
                    spans.append((window_start, window_start + window_size))

            index = end_index + 1

        return spans

    def _token_from_match(self, match: re.Match[str]) -> _Token:
        """Convert a token regex match into a token model."""

        return _Token(text=match.group(0), start=match.start(), end=match.end())

    def _mention_from_tokens(
        self,
        paragraph_id: str,
        tokens: list[_Token],
        kind: str,
    ) -> ConceptMention:
        """Build a mention from token slices."""

        surface = " ".join(token.text for token in tokens)
        return self._make_mention(paragraph_id, surface, kind, tokens[0].start, tokens[-1].end)

    def _make_mention(
        self,
        paragraph_id: str,
        surface: str,
        kind: str,
        start: int,
        end: int,
    ) -> ConceptMention:
        """Create a normalized concept mention."""

        normalized = normalize_surface(surface, lowercase=self._config.lowercase)
        concept_id = self._make_concept_id(normalized, kind)
        return ConceptMention(
            paragraph_id=paragraph_id,
            concept_id=concept_id,
            surface=surface,
            normalized=normalized,
            kind=kind,
            start=start,
            end=end,
        )

    def _make_concept_id(self, normalized: str, kind: str) -> str:
        """Create a stable identifier for a normalized concept."""

        digest = hashlib.sha256(f"{kind}:{normalized}".encode()).hexdigest()
        return digest[:16]

    def _is_entity_like(self, text: str) -> bool:
        """Return whether a token looks like an entity token."""

        return text[:1].isupper() or text.isupper()

    def _is_stopword(self, text: str) -> bool:
        """Return whether a token should terminate a candidate phrase."""

        normalized = normalize_surface(text, lowercase=True)
        return normalized in _STOPWORDS

    def _as_int(self, value: object) -> int:
        """Convert known integer-like values to ``int``."""

        if isinstance(value, int):
            return value
        raise TypeError("spaCy span offsets must be integers")

    def _iter_objects(self, value: object) -> tuple[object, ...]:
        """Convert an arbitrary iterable object into a typed tuple boundary."""

        if isinstance(value, Iterable):
            return tuple(cast(Iterable[object], value))
        return ()
