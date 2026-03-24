"""Typed public configuration for the label generation pipeline."""

from dataclasses import dataclass, field
from typing import Literal

ExtractorMode = Literal["spacy", "heuristic", "llm"]
LLMProviderName = Literal["openai", "mistral", "qwen"]


@dataclass(slots=True)
class LLMExtractionConfig:
    """Configuration for provider-backed LLM concept extraction.

    Attributes:
        provider: LLM provider identifier.
        model: Provider model name.
        api_key_env_var: Optional environment variable holding the API key.
        base_url: Optional base URL override for the provider endpoint.
        organization: Optional organization or tenant identifier.
        timeout_seconds: Request timeout for one LLM call.
        max_retries: Number of retries for transient provider failures.
        temperature: Sampling temperature for concept extraction.
        max_output_tokens: Maximum completion length per batch response.
        batch_size: Number of paragraphs sent in one provider request.
        max_concepts_per_paragraph: Hard cap for parsed concepts per paragraph.
        cache_enabled: Whether to cache parsed paragraph concepts on disk.
        cache_dir: Cache directory for parsed LLM extraction outputs.
        prompt_version: Prompt version identifier that participates in cache keys.
        prompt_template: Optional user prompt template override. When unset, the
            built-in release prompt is used.
    """

    provider: LLMProviderName = "openai"
    model: str = ""
    api_key_env_var: str | None = None
    base_url: str | None = None
    organization: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0
    max_output_tokens: int = 512
    batch_size: int = 8
    max_concepts_per_paragraph: int = 12
    cache_enabled: bool = True
    cache_dir: str | None = ".labelgen-cache"
    prompt_version: str = "v1"
    prompt_template: str | None = None


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
        reject_url_like_concepts: Drop concepts dominated by URLs and support
            links.
        reject_generic_shell_concepts: Drop generic support-document phrases and
            pronoun-like shells that are not useful labels.
        merge_concepts_by_normalized_text: Merge mentions with identical
            normalized text even when extracted with different kinds.
        clean_technical_documents: Apply support-document cleanup before concept
            extraction.
        strip_urls: Remove raw URLs from cleaned paragraph text when technical
            document cleanup is enabled.
        suppress_section_headers: Remove common support-note section headers when
            technical document cleanup is enabled.
        llm: Provider-backed LLM extraction configuration used when the top-level
            extractor mode is set to `llm`.
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
    reject_url_like_concepts: bool = True
    reject_generic_shell_concepts: bool = True
    merge_concepts_by_normalized_text: bool = True
    clean_technical_documents: bool = True
    strip_urls: bool = True
    suppress_section_headers: bool = True
    llm: LLMExtractionConfig = field(default_factory=LLMExtractionConfig)
    spacy_model_name: str = "en_core_web_sm"
    allowed_kinds: tuple[str, ...] = ("entity", "noun_phrase", "llm_concept")


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
        extractor_mode: Preferred extraction mode. When unset, the deprecated
            `use_nlp_extractor` flag is used for compatibility.
        use_nlp_extractor: Deprecated compatibility flag. `True` maps to the
            spaCy extractor and `False` maps to the deterministic heuristic extractor.
        use_graph_community_detection: Use Leiden community detection when `True`,
            otherwise use the deterministic connected-components detector.
    """

    random_seed: int = 42
    extractor_mode: ExtractorMode | None = None
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

    def resolved_extractor_mode(self) -> ExtractorMode:
        """Resolve the active extractor mode with backward compatibility."""

        if self.extractor_mode is not None:
            return self.extractor_mode
        return "spacy" if self.use_nlp_extractor else "heuristic"
