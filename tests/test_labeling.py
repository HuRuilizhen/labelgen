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
