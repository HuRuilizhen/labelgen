"""Deterministic connected-components community detection."""

from __future__ import annotations

from collections import defaultdict

from labelgen.community.detector import CommunityDetector
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.types import Community


class ConnectedComponentsCommunityDetector(CommunityDetector):
    """Fallback community detector based on graph connected components.

    This detector is used when graph-based community detection is explicitly disabled.
    It keeps deterministic ordering and stable community identifiers.
    """

    def detect(self, graph: ConceptGraph) -> list[Community]:
        """Group concept nodes by connected components."""

        if not graph.node_ids:
            return []

        adjacency: dict[str, set[str]] = {node_id: set() for node_id in graph.node_ids}
        for left, right in sorted(graph.edge_weights):
            adjacency[left].add(right)
            adjacency[right].add(left)

        memberships: dict[str, int] = {}
        component_index = 0
        for node_id in sorted(graph.node_ids):
            if node_id in memberships:
                continue

            stack = [node_id]
            while stack:
                current = stack.pop()
                if current in memberships:
                    continue
                memberships[current] = component_index
                stack.extend(sorted(adjacency[current], reverse=True))
            component_index += 1

        grouped: dict[int, list[str]] = defaultdict(list)
        for node_id in graph.node_ids:
            grouped[memberships[node_id]].append(node_id)

        sorted_groups = sorted(
            (sorted(concept_ids) for concept_ids in grouped.values()),
            key=lambda concept_ids: (concept_ids[0], len(concept_ids)),
        )

        return [
            Community(
                id=f"community-{index}",
                concept_ids=concept_ids,
                display_name=f"community-{index}",
                representative_concepts=concept_ids[:5],
                size=len(concept_ids),
            )
            for index, concept_ids in enumerate(sorted_groups)
        ]
