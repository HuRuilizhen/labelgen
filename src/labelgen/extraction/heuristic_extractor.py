"""Deterministic heuristic concept extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

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


class HeuristicConceptExtractor(ConceptExtractor):
    """Deterministic regex-based extractor used when NLP extraction is disabled.

    This extractor treats contiguous capitalized tokens as lightweight entities and
    contiguous non-stopword spans as candidate noun phrases.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config

    def extract(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract concept mentions from normalized paragraphs."""

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
