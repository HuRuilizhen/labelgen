"""Tests for community verbalization."""

from labelgen.config import VerbalizationConfig
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.labeling.verbalizer import verbalize_communities
from labelgen.types import Community, Concept


def test_verbalizer_prefers_higher_document_frequency_then_degree() -> None:
    concepts = [
        Concept(
            id="c-low",
            surface="gamma",
            normalized="gamma",
            kind="noun_phrase",
            document_frequency=1,
        ),
        Concept(
            id="c-high",
            surface="alpha",
            normalized="alpha",
            kind="noun_phrase",
            document_frequency=3,
        ),
        Concept(
            id="c-mid",
            surface="beta",
            normalized="beta",
            kind="noun_phrase",
            document_frequency=2,
        ),
    ]
    communities = [
        Community(
            id="community-0",
            concept_ids=["c-low", "c-high", "c-mid"],
            display_name="community-0",
            representative_concepts=[],
            size=3,
        )
    ]
    graph = ConceptGraph(
        node_ids=["c-low", "c-high", "c-mid"],
        edge_weights={
            ("c-high", "c-mid"): 4,
            ("c-low", "c-mid"): 1,
        },
    )

    result = verbalize_communities(communities, concepts, graph, VerbalizationConfig())

    assert result[0].representative_concepts[:3] == ["alpha", "beta", "gamma"]
    assert result[0].display_name == "alpha / beta / gamma"
