"""Typed public configuration for the label generation pipeline."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ExtractionConfig:
    """Configuration for concept extraction behavior.

    Attributes:
        lowercase: Normalize extracted concept text to lowercase.
        min_concept_length: Drop concepts shorter than this normalized length.
        min_document_frequency: Minimum training-corpus document frequency required
            for a concept to be retained in the fitted model.
        max_concept_df_ratio: Maximum training-corpus document-frequency ratio allowed
            before a concept is treated as too common.
        max_phrase_length: Maximum heuristic phrase length used by the fallback extractor.
        reject_stopword_concepts: Drop concepts made entirely of stopwords.
        spacy_model_name: Installed spaCy pipeline name used by the default NLP
            extractor. `en_core_web_sm` is the recommended default, but callers
            can point this to another compatible installed pipeline.
        allowed_kinds: Allowed concept kinds after extraction.
    """

    lowercase: bool = True
    min_concept_length: int = 2
    min_document_frequency: int = 1
    max_concept_df_ratio: float = 1.0
    max_phrase_length: int = 4
    reject_stopword_concepts: bool = True
    spacy_model_name: str = "en_core_web_sm"
    allowed_kinds: tuple[str, ...] = ("entity", "noun_phrase")


@dataclass(slots=True)
class GraphConfig:
    """Configuration for concept co-occurrence graph construction."""

    min_edge_weight: int = 1


@dataclass(slots=True)
class CommunityDetectionConfig:
    """Configuration for graph community detection."""

    resolution: float = 1.0
    random_seed: int = 42


@dataclass(slots=True)
class LabelAssignmentConfig:
    """Configuration for paragraph-to-label assignment.

    Attributes:
        max_labels_per_paragraph: Maximum number of labels returned per paragraph.
        min_evidence_concepts: Minimum number of distinct evidence concepts needed.
        min_label_support: Minimum label score required for a label assignment.
    """

    max_labels_per_paragraph: int = 3
    min_evidence_concepts: int = 1
    min_label_support: float = 1.0


@dataclass(slots=True)
class VerbalizationConfig:
    """Configuration for human-readable community naming."""

    top_k_label_terms: int = 5
    max_display_terms: int = 3


@dataclass(slots=True)
class LabelGeneratorConfig:
    """Top-level public configuration for `LabelGenerator`.

    Attributes:
        random_seed: Global deterministic seed used by graph algorithms.
        use_nlp_extractor: Use the default spaCy-backed extractor when `True`,
            otherwise use the deterministic heuristic extractor.
        use_graph_community_detection: Use Leiden community detection when `True`,
            otherwise use the deterministic connected-components detector.
    """

    random_seed: int = 42
    use_nlp_extractor: bool = True
    use_graph_community_detection: bool = True
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    community_detection: CommunityDetectionConfig = field(
        default_factory=CommunityDetectionConfig
    )
    label_assignment: LabelAssignmentConfig = field(
        default_factory=LabelAssignmentConfig
    )
    verbalization: VerbalizationConfig = field(default_factory=VerbalizationConfig)
