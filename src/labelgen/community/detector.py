"""Abstract interface for community detection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from labelgen.graph.concept_graph import ConceptGraph
from labelgen.types import Community


class CommunityDetector(ABC):
    """Abstract base class for community detection implementations."""

    @abstractmethod
    def detect(self, graph: ConceptGraph) -> list[Community]:
        """Detect communities in a concept graph."""
