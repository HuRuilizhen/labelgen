"""Helpers for human-readable community labels."""

from labelgen.types import Community, Concept


def verbalize_communities(
    communities: list[Community],
    concepts: list[Concept],
) -> list[Community]:
    """Create readable labels from representative community concepts."""

    concepts_by_id = {concept.id: concept for concept in concepts}
    verbalized: list[Community] = []
    for community in communities:
        representative_concepts = [
            concepts_by_id[concept_id].normalized
            for concept_id in community.concept_ids[:5]
            if concept_id in concepts_by_id
        ]
        display_name = ", ".join(representative_concepts[:3]) or community.id
        verbalized.append(
            Community(
                id=community.id,
                concept_ids=community.concept_ids,
                display_name=display_name,
                representative_concepts=representative_concepts,
                size=community.size,
            )
        )
    return verbalized
