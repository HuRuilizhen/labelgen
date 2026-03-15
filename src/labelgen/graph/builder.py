"""Concept graph construction helpers."""

from __future__ import annotations

from collections import Counter, defaultdict

from labelgen.config import GraphConfig
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.types import ConceptMention


def build_concept_graph(mentions: list[ConceptMention], config: GraphConfig) -> ConceptGraph:
    """Build a simple concept co-occurrence graph from paragraph mentions."""

    paragraph_to_concepts: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        paragraph_to_concepts[mention.paragraph_id].add(mention.concept_id)

    edge_weights: Counter[tuple[str, str]] = Counter()
    for concept_ids in paragraph_to_concepts.values():
        sorted_ids = sorted(concept_ids)
        for index, left in enumerate(sorted_ids):
            for right in sorted_ids[index + 1 :]:
                edge_weights[(left, right)] += 1

    filtered_edges = {
        edge: weight for edge, weight in edge_weights.items() if weight >= config.min_edge_weight
    }
    node_ids = sorted({mention.concept_id for mention in mentions})
    return ConceptGraph(node_ids=node_ids, edge_weights=filtered_edges)
