"""Helpers for filtering extracted concepts."""

import re

from labelgen.config import ExtractionConfig
from labelgen.types import ConceptMention

_ALNUM_RE = re.compile(r"[a-z0-9]")


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
        if config.reject_stopword_concepts and _is_all_stopwords(mention.normalized):
            continue
        filtered.append(mention)
    return filtered


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
