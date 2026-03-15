"""Helpers for filtering extracted concepts."""

from labelgen.config import ExtractionConfig
from labelgen.types import ConceptMention


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
        filtered.append(mention)
    return filtered
