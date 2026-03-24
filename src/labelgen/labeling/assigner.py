"""Paragraph label assignment helpers."""

from __future__ import annotations

from collections import defaultdict

from labelgen.config import LabelAssignmentConfig
from labelgen.extraction.filtering import is_noisy_concept_text
from labelgen.types import Community, ConceptMention, Paragraph, ParagraphLabels


def assign_paragraph_labels(
    paragraphs: list[Paragraph],
    mentions: list[ConceptMention],
    communities: list[Community],
    config: LabelAssignmentConfig,
) -> list[ParagraphLabels]:
    """Assign labels to paragraphs based on concept-community membership."""

    concept_to_community: dict[str, str] = {}
    community_by_id = {community.id: community for community in communities}
    for community in communities:
        for concept_id in community.concept_ids:
            concept_to_community[concept_id] = community.id

    paragraph_evidence_by_label: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for mention in mentions:
        community_id = concept_to_community.get(mention.concept_id)
        if community_id is None:
            continue
        paragraph_evidence_by_label[mention.paragraph_id][community_id].add(mention.concept_id)

    labels: list[ParagraphLabels] = []
    for paragraph in paragraphs:
        evidence_by_label = paragraph_evidence_by_label.get(paragraph.id, {})
        ranked = sorted(
            (
                (
                    community_id,
                    float(len(concept_ids))
                    * _community_quality_weight(community_by_id[community_id]),
                )
                for community_id, concept_ids in evidence_by_label.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        ranked = ranked[: config.max_labels_per_paragraph]
        minimum_support = max(float(config.min_evidence_concepts), config.min_label_support)
        label_ids = [
            community_id
            for community_id, score in ranked
            if score >= minimum_support
        ]
        selected_evidence = sorted(
            {
                concept_id
                for community_id in label_ids
                for concept_id in evidence_by_label.get(community_id, set())
            }
        )
        labels.append(
            ParagraphLabels(
                paragraph_id=paragraph.id,
                label_ids=label_ids,
                evidence_concept_ids=selected_evidence,
                label_scores={
                    community_id: score
                    for community_id, score in ranked
                },
            )
        )
    return labels


def _community_quality_weight(community: Community) -> float:
    """Return a ranking weight for a community based on label quality."""

    representative_concepts = community.representative_concepts[:3]
    if not representative_concepts:
        return 1.0

    noisy_count = sum(1 for concept in representative_concepts if is_noisy_concept_text(concept))
    weight = 1.0
    if noisy_count == len(representative_concepts):
        weight *= 0.35
    elif noisy_count > 0:
        weight *= 0.6

    if noisy_count > 0:
        if community.size >= 100:
            weight *= 0.7
        elif community.size >= 60:
            weight *= 0.85
    return weight
