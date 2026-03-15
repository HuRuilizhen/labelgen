"""Simple in-memory concept graph representation."""

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
