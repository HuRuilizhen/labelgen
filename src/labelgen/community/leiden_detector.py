"""Leiden community detection with deterministic fallback behavior."""

from __future__ import annotations

import importlib
from collections import defaultdict
from typing import cast

from labelgen.community.detector import CommunityDetector
from labelgen.config import CommunityDetectionConfig
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.types import Community


class LeidenCommunityDetector(CommunityDetector):
    """Community detector implementation based on Leiden."""

    def __init__(self, config: CommunityDetectionConfig) -> None:
        self._config = config

    def detect(self, graph: ConceptGraph) -> list[Community]:
        """Detect communities in the graph."""

        if not graph.node_ids:
            return []

        memberships = self._detect_with_leiden(graph)
        if memberships is None:
            memberships = self._detect_with_connected_components(graph)
        return self._build_communities(graph.node_ids, memberships)

    def _detect_with_leiden(self, graph: ConceptGraph) -> list[int] | None:
        """Run Leiden if optional graph dependencies are available."""

        try:
            igraph = importlib.import_module("igraph")
            leidenalg = importlib.import_module("leidenalg")
        except ImportError:
            return None

        node_to_index = {node_id: index for index, node_id in enumerate(graph.node_ids)}
        sorted_edges = sorted(graph.edge_weights)
        edge_list = [(node_to_index[left], node_to_index[right]) for left, right in sorted_edges]
        weights = [graph.edge_weights[edge] for edge in sorted_edges]

        graph_object = igraph.Graph(n=len(graph.node_ids), edges=edge_list, directed=False)
        partition_type = getattr(leidenalg, "RBConfigurationVertexPartition", None)
        find_partition = getattr(leidenalg, "find_partition", None)
        if partition_type is None or not callable(find_partition):
            return None

        partition = find_partition(
            graph_object,
            partition_type,
            weights=weights if weights else None,
            seed=self._config.random_seed,
            resolution_parameter=self._config.resolution,
        )
        membership = self._coerce_membership(getattr(partition, "membership", None))
        if membership is None or len(membership) != len(graph.node_ids):
            return None
        return membership

    def _detect_with_connected_components(self, graph: ConceptGraph) -> list[int]:
        """Compute deterministic connected components as a fallback."""

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

        return [memberships[node_id] for node_id in graph.node_ids]

    def _build_communities(self, node_ids: list[str], membership: list[int]) -> list[Community]:
        """Build stable community models from membership assignments."""

        grouped: dict[int, list[str]] = defaultdict(list)
        for node_id, community_index in zip(node_ids, membership, strict=True):
            grouped[community_index].append(node_id)

        sorted_groups = sorted(
            (sorted(concept_ids) for concept_ids in grouped.values()),
            key=lambda concept_ids: (concept_ids[0], len(concept_ids)),
        )

        communities: list[Community] = []
        for index, concept_ids in enumerate(sorted_groups):
            communities.append(
                Community(
                    id=f"community-{index}",
                    concept_ids=concept_ids,
                    display_name=f"community-{index}",
                    representative_concepts=concept_ids[:5],
                    size=len(concept_ids),
                )
            )
        return communities

    def _coerce_membership(self, value: object) -> list[int] | None:
        """Convert a partition membership object into a typed list."""

        if not isinstance(value, list):
            return None

        membership: list[int] = []
        for item in cast(list[object], value):
            if not isinstance(item, int):
                return None
            membership.append(item)
        return membership
