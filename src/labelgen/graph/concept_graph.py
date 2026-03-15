"""Simple in-memory concept graph representation."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class ConceptGraph:
    """Graph of co-occurring concepts."""

    node_ids: list[str]
    edge_weights: dict[tuple[str, str], int]

    @property
    def edge_count(self) -> int:
        """Return the number of edges."""

        return len(self.edge_weights)

    @property
    def node_count(self) -> int:
        """Return the number of nodes."""

        return len(self.node_ids)

    def weighted_degree_map(self) -> dict[str, int]:
        """Return weighted degree scores for graph nodes."""

        scores: dict[str, int] = defaultdict(int)
        for (left, right), weight in self.edge_weights.items():
            scores[left] += weight
            scores[right] += weight
        return {node_id: scores[node_id] for node_id in self.node_ids}
