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
- `use_nlp_extractor`: use spaCy extraction when `True`
- `use_graph_community_detection`: use Leiden community detection when `True`
- `extraction`: `ExtractionConfig`
- `graph`: `GraphConfig`
- `community_detection`: `CommunityDetectionConfig`
- `label_assignment`: `LabelAssignmentConfig`
- `verbalization`: `VerbalizationConfig`

## ExtractionConfig

Key extraction options:

- `min_document_frequency`
- `max_concept_df_ratio`
- `reject_stopword_concepts`
- `reject_url_like_concepts`
- `reject_generic_shell_concepts`
- `merge_concepts_by_normalized_text`
- `clean_technical_documents`
- `strip_urls`
- `suppress_section_headers`
- `spacy_model_name`

`spacy_model_name` defaults to `en_core_web_sm`, which is the recommended
default model for the public pipeline.

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

By default:

- spaCy is used for concept extraction
- Leiden is used for community detection

You can explicitly opt out:

```python
from labelgen import LabelGenerator, LabelGeneratorConfig

config = LabelGeneratorConfig(
    use_nlp_extractor=False,
    use_graph_community_detection=False,
)
generator = LabelGenerator(config)
```

When `use_nlp_extractor=True`, the configured spaCy model must already be
installed. Missing models raise an explicit runtime error.
