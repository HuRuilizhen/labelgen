"""Leiden community detection."""

from __future__ import annotations

import importlib
from typing import cast

from labelgen.community.detector import CommunityDetector
from labelgen.config import CommunityDetectionConfig
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.types import Community


class LeidenCommunityDetector(CommunityDetector):
    """Community detector implementation backed by Leiden.

    This is the default detector for `0.1.0`. If `igraph` or `leidenalg` is missing,
    initialization or detection raises a clear runtime error so the caller can
    explicitly opt out to the connected-components detector.
    """

    def __init__(self, config: CommunityDetectionConfig) -> None:
        self._config = config

    def detect(self, graph: ConceptGraph) -> list[Community]:
        """Detect communities with Leiden and return stable community models."""

        if not graph.node_ids:
            return []

        memberships = self._detect_with_leiden(graph)
        return self._build_communities(graph.node_ids, memberships)

    def _detect_with_leiden(self, graph: ConceptGraph) -> list[int]:
        """Run Leiden community detection and return node memberships."""

        try:
            igraph = importlib.import_module("igraph")
            leidenalg = importlib.import_module("leidenalg")
        except ImportError as error:
            raise RuntimeError(
                "igraph and leidenalg are required for graph community detection. "
                "Disable graph community detection in LabelGeneratorConfig to use "
                "the connected-components detector."
            ) from error

        node_to_index = {node_id: index for index, node_id in enumerate(graph.node_ids)}
        sorted_edges = sorted(graph.edge_weights)
        edge_list = [(node_to_index[left], node_to_index[right]) for left, right in sorted_edges]
        weights = [graph.edge_weights[edge] for edge in sorted_edges]

        graph_object = igraph.Graph(n=len(graph.node_ids), edges=edge_list, directed=False)
        partition_type = getattr(leidenalg, "RBConfigurationVertexPartition", None)
        find_partition = getattr(leidenalg, "find_partition", None)
        if partition_type is None or not callable(find_partition):
            raise RuntimeError("leidenalg.find_partition is unavailable in the installed package.")

        partition = find_partition(
            graph_object,
            partition_type,
            weights=weights if weights else None,
            seed=self._config.random_seed,
            resolution_parameter=self._config.resolution,
        )
        membership = self._coerce_membership(getattr(partition, "membership", None))
        if membership is None or len(membership) != len(graph.node_ids):
            raise RuntimeError("Leiden partition did not return a valid membership list.")
        return membership

    def _build_communities(self, node_ids: list[str], membership: list[int]) -> list[Community]:
        """Build stable community models from membership assignments."""

        from collections import defaultdict

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
