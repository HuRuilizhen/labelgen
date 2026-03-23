"""Typed public result models for the label generation pipeline."""

from dataclasses import dataclass, field
from typing import Any


def _empty_string_list() -> list[str]:
    """Create an empty string list with an explicit type."""

    return []


@dataclass(slots=True)
class Paragraph:
    """A normalized paragraph accepted by and returned from the pipeline."""

    id: str
    text: str
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class Concept:
    """A normalized concept retained by the fitted label-generation pipeline."""

    id: str
    surface: str
    normalized: str
    kind: str
    document_frequency: int | None = None


@dataclass(slots=True)
class ConceptMention:
    """An extracted concept mention with optional character offsets."""

    paragraph_id: str
    concept_id: str
    surface: str
    normalized: str
    kind: str
    start: int | None = None
    end: int | None = None


@dataclass(slots=True)
class Community:
    """A detected concept community with human-readable naming metadata."""

    id: str
    concept_ids: list[str]
    display_name: str
    representative_concepts: list[str]
    size: int


@dataclass(slots=True)
class ParagraphLabels:
    """Assigned labels and evidence scores for a single paragraph."""

    paragraph_id: str
    label_ids: list[str]
    evidence_concept_ids: list[str] = field(default_factory=_empty_string_list)
    label_scores: dict[str, float] = field(default_factory=lambda: {})


@dataclass(slots=True)
class GraphSummary:
    """Basic statistics and debug metadata for the concept graph."""

    node_count: int
    edge_count: int
    metadata: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(slots=True)
class LabelGenerationResult:
    """End-to-end structured result returned by `fit_transform` and `transform`."""

    paragraphs: list[Paragraph]
    concepts: list[Concept]
    mentions: list[ConceptMention]
    communities: list[Community]
    paragraph_labels: list[ParagraphLabels]
    graph_summary: GraphSummary | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: {})
