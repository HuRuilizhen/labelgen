"""Tests for concept graph construction."""

from labelgen.config import GraphConfig
from labelgen.graph.builder import build_concept_graph
from labelgen.types import ConceptMention


def test_graph_cooccurrence_is_not_inflated_by_repeated_mentions() -> None:
    mentions = [
        ConceptMention(
            paragraph_id="p1",
            concept_id="c1",
            surface="alpha",
            normalized="alpha",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c1",
            surface="alpha",
            normalized="alpha",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p1",
            concept_id="c2",
            surface="beta",
            normalized="beta",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p2",
            concept_id="c1",
            surface="alpha",
            normalized="alpha",
            kind="noun_phrase",
        ),
        ConceptMention(
            paragraph_id="p2",
            concept_id="c2",
            surface="beta",
            normalized="beta",
            kind="noun_phrase",
        ),
    ]

    graph = build_concept_graph(mentions, GraphConfig())

    assert graph.edge_weights == {("c1", "c2"): 2}
