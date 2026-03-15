"""Helpers for human-readable community labels."""

from labelgen.config import VerbalizationConfig
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.types import Community, Concept


def verbalize_communities(
    communities: list[Community],
    concepts: list[Concept],
    graph: ConceptGraph,
    config: VerbalizationConfig,
) -> list[Community]:
    """Create readable labels from representative community concepts."""

    concepts_by_id = {concept.id: concept for concept in concepts}
    weighted_degree = graph.weighted_degree_map()

    verbalized: list[Community] = []
    for community in communities:
        ranked_concepts = sorted(
            (
                concepts_by_id[concept_id]
                for concept_id in community.concept_ids
                if concept_id in concepts_by_id
            ),
            key=lambda concept: (
                -float(concept.document_frequency or 0),
                -float(weighted_degree.get(concept.id, 0)),
                concept.normalized,
            ),
        )
        representative_concepts = [
            concept.normalized for concept in ranked_concepts[: config.top_k_label_terms]
        ]
        display_name = (
            " / ".join(representative_concepts[: config.max_display_terms]) or community.id
        )
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
