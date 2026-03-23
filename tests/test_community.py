"""Tests for community detection behavior."""

from labelgen.community.connected_components_detector import (
    ConnectedComponentsCommunityDetector,
)
from labelgen.community.leiden_detector import LeidenCommunityDetector
from labelgen.config import CommunityDetectionConfig
from labelgen.graph.concept_graph import ConceptGraph


def test_connected_components_detector_groups_disconnected_subgraphs() -> None:
    detector = ConnectedComponentsCommunityDetector()
    graph = ConceptGraph(
        node_ids=["a", "b", "c", "d"],
        edge_weights={
            ("a", "b"): 2,
            ("c", "d"): 2,
        },
    )

    communities = detector.detect(graph)

    assert [community.concept_ids for community in communities] == [["a", "b"], ["c", "d"]]


def test_detect_returns_two_communities_for_disconnected_graph() -> None:
    detector = LeidenCommunityDetector(CommunityDetectionConfig())
    graph = ConceptGraph(
        node_ids=["a", "b", "c", "d"],
        edge_weights={
            ("a", "b"): 3,
            ("c", "d"): 3,
        },
    )

    communities = detector.detect(graph)

    assert len(communities) == 2
    assert [community.concept_ids for community in communities] == [["a", "b"], ["c", "d"]]
    assert [community.id for community in communities] == ["community-0", "community-1"]
