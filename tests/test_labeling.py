"""Tests for paragraph label assignment."""

from labelgen.config import LabelAssignmentConfig
from labelgen.labeling.assigner import assign_paragraph_labels
from labelgen.types import Community, ConceptMention, Paragraph


def test_assigner_returns_labels_for_all_paragraphs() -> None:
    paragraphs = [
        Paragraph(id="p1", text="Alpha beta"),
        Paragraph(id="p2", text="Gamma"),
    ]
    communities = [
        Community(
            id="community-0",
            concept_ids=["c1"],
            display_name="alpha",
            representative_concepts=["alpha"],
            size=1,
        )
    ]
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="c1",
            surface="Alpha",
            normalized="alpha",
            kind="noun_phrase",
        )
    ]

    result = assign_paragraph_labels(paragraphs, mentions, communities, LabelAssignmentConfig())

    assert [item.paragraph_id for item in result] == ["p1", "p2"]
    assert result[0].label_ids == ["community-0"]
    assert result[1].label_ids == []
    assert result[1].evidence_concept_ids == []


def test_assigner_preserves_input_paragraph_order() -> None:
    paragraphs = [
        Paragraph(id="z-last", text="Alpha"),
        Paragraph(id="a-first", text="Beta"),
    ]
    communities = [
        Community(
            id="community-0",
            concept_ids=["c1", "c2"],
            display_name="alpha / beta",
            representative_concepts=["alpha", "beta"],
            size=2,
        )
    ]
    mentions = [
        ConceptMention(
            paragraph_id="z-last",
            concept_id="c1",
            surface="Alpha",
            normalized="alpha",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="a-first",
            concept_id="c2",
            surface="Beta",
            normalized="beta",
            kind="noun_phrase",
        ),
    ]

    result = assign_paragraph_labels(paragraphs, mentions, communities, LabelAssignmentConfig())

    assert [item.paragraph_id for item in result] == ["z-last", "a-first"]


def test_assigner_deduplicates_repeated_concepts_per_label() -> None:
    paragraphs = [Paragraph(id="p1", text="Alpha alpha beta")]
    communities = [
        Community(
            id="community-0",
            concept_ids=["c1", "c2"],
            display_name="alpha, beta",
            representative_concepts=["alpha", "beta"],
            size=2,
        )
    ]
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="c1",
            surface="Alpha",
            normalized="alpha",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c1",
            surface="Alpha",
            normalized="alpha",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c2",
            surface="beta",
            normalized="beta",
            kind="noun_phrase",
        ),
    ]

    result = assign_paragraph_labels(paragraphs, mentions, communities, LabelAssignmentConfig())

    assert result[0].label_scores == {"community-0": 2.0}
    assert result[0].evidence_concept_ids == ["c1", "c2"]


def test_assigner_respects_min_label_support() -> None:
    paragraphs = [Paragraph(id="p1", text="Alpha beta")]
    communities = [
        Community(
            id="community-0",
            concept_ids=["c1", "c2"],
            display_name="alpha / beta",
            representative_concepts=["alpha", "beta"],
            size=2,
        )
    ]
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="c1",
            surface="Alpha",
            normalized="alpha",
            kind="noun_phrase",
        )
    ]
    config = LabelAssignmentConfig()
    config.min_label_support = 2.0

    result = assign_paragraph_labels(paragraphs, mentions, communities, config)

    assert result[0].label_ids == []


def test_assigner_downweights_oversized_noisy_communities() -> None:
    paragraphs = [Paragraph(id="p1", text="Alpha beta gamma")]
    communities = [
        Community(
            id="generic-community",
            concept_ids=["c1", "c2"],
            display_name="problem summary / www.ibm.com/support / you",
            representative_concepts=["problem summary", "www.ibm.com/support", "you"],
            size=150,
        ),
        Community(
            id="topic-community",
            concept_ids=["c3", "c4"],
            display_name="internal compiler optimization / inlining",
            representative_concepts=["internal compiler optimization", "inlining"],
            size=8,
        ),
    ]
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="c1",
            surface="problem summary",
            normalized="problem summary",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c2",
            surface="www.ibm.com/support",
            normalized="www.ibm.com/support",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c3",
            surface="internal compiler optimization",
            normalized="internal compiler optimization",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c4",
            surface="inlining",
            normalized="inlining",
            kind="noun_phrase",
        ),
    ]

    result = assign_paragraph_labels(paragraphs, mentions, communities, LabelAssignmentConfig())

    assert result[0].label_ids[0] == "topic-community"
    assert result[0].label_scores["topic-community"] > result[0].label_scores["generic-community"]
