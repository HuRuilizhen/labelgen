# Public API

This document summarizes the supported public interface of `paralabelgen`.

## Package Entry Point

Install with:

```bash
pip install paralabelgen
```

Import with:

```python
from labelgen import LabelGenerator, LabelGeneratorConfig
```

## LabelGenerator

`LabelGenerator` is the main stateful pipeline entrypoint.

### Constructor

```python
LabelGenerator(config: LabelGeneratorConfig | None = None)
```

If `config` is omitted, the default configuration is used.

### Methods

#### fit

```python
fit(paragraphs: list[str] | list[Paragraph]) -> LabelGenerator
```

Learns concepts and communities from a training corpus.

#### transform

```python
transform(paragraphs: list[str] | list[Paragraph]) -> LabelGenerationResult
```

Applies previously learned communities to new paragraphs. This method requires
the generator to be fitted first.

#### fit_transform

```python
fit_transform(paragraphs: list[str] | list[Paragraph]) -> LabelGenerationResult
```

Fits the generator and labels the same input in one pass.

#### save

```python
save(path: str | Path) -> None
```

Serializes the generator configuration and fitted state.

#### load

```python
LabelGenerator.load(path: str | Path) -> LabelGenerator
```

Loads a previously serialized generator.

### Properties

- `extractor_name`: active extractor implementation name
- `detector_name`: active community detector implementation name

## LabelGeneratorConfig

`LabelGeneratorConfig` is the top-level configuration object.

### Top-Level Fields

- `random_seed`: deterministic seed used across graph-related steps
- `extractor_mode`: preferred extractor mode, one of `spacy`, `heuristic`, or `llm`
- `use_nlp_extractor`: deprecated compatibility flag; `True` maps to `spacy`
  and `False` maps to `heuristic`
- `use_graph_community_detection`: use Leiden community detection when `True`,
  otherwise use deterministic connected-components detection
- `extraction`: `ExtractionConfig`
- `graph`: `GraphConfig`
- `community_detection`: `CommunityDetectionConfig`
- `label_assignment`: `LabelAssignmentConfig`
- `verbalization`: `VerbalizationConfig`

New code should prefer `extractor_mode` over `use_nlp_extractor`.

## ExtractionConfig

Key extraction options:

- `lowercase`
- `min_concept_length`
- `min_document_frequency`
- `max_concept_df_ratio`
- `max_phrase_length`
- `reject_stopword_concepts`
- `reject_url_like_concepts`
- `reject_generic_shell_concepts`
- `merge_concepts_by_normalized_text`
- `clean_technical_documents`
- `strip_urls`
- `suppress_section_headers`
- `spacy_model_name`
- `llm`

`spacy_model_name` defaults to `en_core_web_sm`, which is the recommended
default model for the public spaCy pipeline.

## LLMExtractionConfig

`ExtractionConfig.llm` controls provider-backed concept extraction.

### Provider And Request Settings

- `provider`: one of `openai`, `mistral`, or `qwen`
- `model`: provider model name
- `api_key_env_var`: optional environment variable override for the API key
- `base_url`: optional provider base URL override
- `organization`: optional organization or tenant identifier
- `timeout_seconds`: request timeout for one provider call
- `max_retries`: retry count for transient request failures
- `temperature`: completion sampling temperature
- `max_output_tokens`: maximum completion length per batch response
- `batch_size`: paragraphs sent in one provider request
- `max_concepts_per_paragraph`: post-parse concept cap per paragraph

### Cache And Artifact Settings

- `cache_enabled`: cache parsed concept lists on disk
- `cache_dir`: cache directory for parsed outputs
- `record_extraction_artifacts`: write structured per-batch artifacts for audit
  and experiment analysis
- `artifact_dir`: artifact output directory

### Prompt Settings

- `prompt_version`: optional human-readable prompt identifier for audit and
  experiment tracking
- `prompt_template`: optional prompt override

Cache invalidation is driven by the effective prompt text, not by
`prompt_version` alone. `prompt_version` is mainly useful as a human-readable
label in artifacts and experiment records.

## Result Models

The package exports these result dataclasses:

- `Paragraph`
- `Concept`
- `ConceptMention`
- `Community`
- `ParagraphLabels`
- `LabelGenerationResult`

These models are returned by `fit_transform()` and `transform()`.

## Serialization Helpers

The package exports:

```python
dump_result(result: LabelGenerationResult, path: str | Path) -> None
load_result(path: str | Path) -> LabelGenerationResult
```

These helpers serialize and reload pipeline results as JSON.

## Default And Fallback Behavior

Default public behavior:

- `extractor_mode="spacy"` uses spaCy concept extraction
- `use_graph_community_detection=True` uses Leiden community detection

Explicit alternatives:

- `extractor_mode="heuristic"` uses the deterministic heuristic extractor
- `extractor_mode="llm"` uses provider-backed LLM extraction
- `use_graph_community_detection=False` uses deterministic connected components

When `extractor_mode="spacy"`, the configured spaCy model must already be
installed. Missing models raise an explicit runtime error.

When `extractor_mode="llm"`, provider configuration must be valid and the
expected API key must be available. The LLM path does not silently fall back to
spaCy or heuristic extraction.

