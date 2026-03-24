"""Helpers for human-readable community labels."""

from labelgen.config import VerbalizationConfig
from labelgen.extraction.filtering import (
    is_generic_shell_concept_text,
    is_url_like_concept_text,
)
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
                _concept_noise_rank(concept.normalized),
                -float(concept.document_frequency or 0),
                -float(weighted_degree.get(concept.id, 0)),
                -len(concept.normalized),
                concept.normalized,
            ),
        )
        representative_concepts = _select_representative_concepts(
            ranked_concepts,
            config.top_k_label_terms,
        )
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


def _concept_noise_rank(normalized: str) -> int:
    """Return a stable rank penalty for low-quality concept text."""

    if is_url_like_concept_text(normalized):
        return 2
    if is_generic_shell_concept_text(normalized):
        return 1
    return 0


def _select_representative_concepts(
    ranked_concepts: list[Concept],
    limit: int,
) -> list[str]:
    """Select representative concepts while preferring content-bearing terms."""

    preferred: list[str] = []
    deferred: list[str] = []
    seen: set[str] = set()

    for concept in ranked_concepts:
        if concept.normalized in seen:
            continue
        seen.add(concept.normalized)
        target = deferred if _concept_noise_rank(concept.normalized) > 0 else preferred
        target.append(concept.normalized)

    return (preferred + deferred)[:limit]
