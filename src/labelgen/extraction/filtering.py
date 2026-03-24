"""Helpers for filtering extracted concepts."""

from __future__ import annotations

import hashlib
import re

from labelgen.config import ExtractionConfig
from labelgen.types import ConceptMention

_ALNUM_RE = re.compile(r"[a-z0-9]")
_URL_LIKE_RE = re.compile(
    r"(?:https?://|www\.|\.com\b|\.org\b|uid=|cgi-bin|docview\.wss|support/)",
    re.IGNORECASE,
)
_GENERIC_SHELLS = frozenset(
    {
        "you",
        "they",
        "one",
        "line",
        "problem summary",
        "problem description",
        "problem conclusion",
        "error description",
        "fix information",
        "apar status",
        "users affected",
        "all active apars",
        "this component",
        "this apar",
        "the problem",
        "the fix",
        "reported component name",
        "fixed component name",
        "direct links",
        "references",
    }
)
_GENERIC_PREFIXES = (
    "reported component name ",
    "fixed component name ",
    "problem summary ",
    "problem description ",
    "problem conclusion ",
    "error description ",
    "fix information ",
    "users affected ",
)


def filter_mentions(
    mentions: list[ConceptMention], config: ExtractionConfig
) -> list[ConceptMention]:
    """Filter mentions using conservative baseline rules."""

    filtered: list[ConceptMention] = []
    for mention in mentions:
        if mention.kind not in config.allowed_kinds:
            continue
        if len(mention.normalized) < config.min_concept_length:
            continue
        if not _ALNUM_RE.search(mention.normalized):
            continue
        if config.reject_url_like_concepts and _is_url_like(mention.normalized):
            continue
        if config.reject_stopword_concepts and _is_all_stopwords(mention.normalized):
            continue
        if config.reject_generic_shell_concepts and _is_generic_shell(mention.normalized):
            continue
        filtered.append(mention)
    return filtered


def canonicalize_mentions(
    mentions: list[ConceptMention],
    config: ExtractionConfig,
) -> list[ConceptMention]:
    """Canonicalize mention concept identifiers after filtering."""

    if not config.merge_concepts_by_normalized_text:
        return mentions

    canonical_ids: dict[str, str] = {}
    canonicalized: list[ConceptMention] = []
    for mention in mentions:
        canonical_id = canonical_ids.setdefault(
            mention.normalized,
            _canonical_concept_id(mention.normalized),
        )
        canonicalized.append(
            ConceptMention(
                paragraph_id=mention.paragraph_id,
                concept_id=canonical_id,
                surface=mention.surface,
                normalized=mention.normalized,
                kind=mention.kind,
                start=mention.start,
                end=mention.end,
            )
        )
    return canonicalized


def is_url_like_concept_text(text: str) -> bool:
    """Return whether concept text is dominated by URLs or support-link syntax."""

    return _is_url_like(text)


def is_generic_shell_concept_text(text: str) -> bool:
    """Return whether concept text is a generic support-document shell."""

    return _is_generic_shell(text)


def is_noisy_concept_text(text: str) -> bool:
    """Return whether concept text should be treated as low-quality label text."""

    return _is_url_like(text) or _is_generic_shell(text)


def _is_all_stopwords(text: str) -> bool:
    """Return whether all tokens in a normalized concept are stopwords."""

    stopwords = {
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
    tokens = [token for token in text.split(" ") if token]
    return bool(tokens) and all(token in stopwords for token in tokens)


def _is_url_like(text: str) -> bool:
    """Return whether a concept is dominated by URLs or support-link syntax."""

    return bool(_URL_LIKE_RE.search(text))


def _is_generic_shell(text: str) -> bool:
    """Return whether a concept is a generic support-document shell."""

    if text in _GENERIC_SHELLS:
        return True
    return any(text.startswith(prefix) for prefix in _GENERIC_PREFIXES)


def _canonical_concept_id(normalized: str) -> str:
    """Build a stable canonical concept identifier from normalized text."""

    digest = hashlib.sha256(f"concept:{normalized}".encode()).hexdigest()
    return digest[:16]
