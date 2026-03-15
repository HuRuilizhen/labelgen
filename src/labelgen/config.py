"""Configuration models for label generation."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ExtractionConfig:
    """Configuration for concept extraction."""

    lowercase: bool = True
    min_concept_length: int = 2
    min_document_frequency: int = 1
    max_concept_df_ratio: float = 1.0
    max_phrase_length: int = 4
    reject_stopword_concepts: bool = True
    prefer_spacy: bool = True
    spacy_model_name: str = "en_core_web_sm"
    allowed_kinds: tuple[str, ...] = ("entity", "noun_phrase")


@dataclass(slots=True)
class GraphConfig:
    """Configuration for concept graph construction."""

    min_edge_weight: int = 1


@dataclass(slots=True)
class CommunityDetectionConfig:
    """Configuration for community detection."""

    resolution: float = 1.0
    random_seed: int = 42


@dataclass(slots=True)
class LabelAssignmentConfig:
    """Configuration for paragraph-to-label assignment."""

    max_labels_per_paragraph: int = 3
    min_evidence_concepts: int = 1
    min_label_support: float = 1.0


@dataclass(slots=True)
class VerbalizationConfig:
    """Configuration for human-readable label naming."""

    top_k_label_terms: int = 5
    max_display_terms: int = 3


@dataclass(slots=True)
class LabelGeneratorConfig:
    """Top-level configuration for the label generation pipeline."""

    random_seed: int = 42
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    community_detection: CommunityDetectionConfig = field(default_factory=CommunityDetectionConfig)
    label_assignment: LabelAssignmentConfig = field(default_factory=LabelAssignmentConfig)
    verbalization: VerbalizationConfig = field(default_factory=VerbalizationConfig)
