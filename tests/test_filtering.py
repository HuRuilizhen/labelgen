"""Tests for concept filtering and canonicalization."""

from labelgen.config import ExtractionConfig
from labelgen.extraction.filtering import canonicalize_mentions, filter_mentions
from labelgen.types import ConceptMention


def test_filter_mentions_rejects_url_like_and_generic_shell_concepts() -> None:
    config = ExtractionConfig()
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="c-url",
            surface="https://example.com/doc",
            normalized="https://example.com/doc",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c-shell",
            surface="Problem Description",
            normalized="problem description",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c-good",
            surface="internal compiler optimization",
            normalized="internal compiler optimization",
            kind="noun_phrase",
        ),
    ]

    filtered = filter_mentions(mentions, config)

    assert [mention.normalized for mention in filtered] == ["internal compiler optimization"]


def test_filter_mentions_rejects_markup_heavy_concepts() -> None:
    config = ExtractionConfig()
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="c-markup",
            surface="[ qradar vulnerability manager",
            normalized="[ qradar vulnerability manager",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c-good",
            surface="qradar vulnerability manager",
            normalized="qradar vulnerability manager",
            kind="noun_phrase",
        ),
    ]

    filtered = filter_mentions(mentions, config)

    assert [mention.normalized for mention in filtered] == ["qradar vulnerability manager"]


def test_canonicalize_mentions_merges_matching_normalized_text_across_kinds() -> None:
    config = ExtractionConfig()
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="entity-id",
            surface="OpenAI",
            normalized="openai",
            kind="entity",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="noun-id",
            surface="OpenAI",
            normalized="openai",
            kind="noun_phrase",
        ),
    ]

    canonicalized = canonicalize_mentions(mentions, config)

    assert canonicalized[0].concept_id == canonicalized[1].concept_id
    assert canonicalized[0].normalized == canonicalized[1].normalized == "openai"


def test_canonicalize_mentions_can_be_disabled() -> None:
    config = ExtractionConfig(merge_concepts_by_normalized_text=False)
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="entity-id",
            surface="OpenAI",
            normalized="openai",
            kind="entity",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="noun-id",
            surface="OpenAI",
            normalized="openai",
            kind="noun_phrase",
        ),
    ]

    canonicalized = canonicalize_mentions(mentions, config)

    assert [mention.concept_id for mention in canonicalized] == ["entity-id", "noun-id"]
