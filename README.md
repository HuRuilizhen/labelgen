# paralabelgen

`paralabelgen` is a Python library for generating discrete multi-label
annotations for text paragraphs from concept extraction, graph communities,
and interpretable assignment rules.

- PyPI distribution: `paralabelgen`
- Python import package: `labelgen`
- Repository: `https://github.com/HuRuilizhen/labelgen`

## Install

```bash
pip install paralabelgen
python -m spacy download en_core_web_sm
```

`en_core_web_sm` is the recommended default model. If you already use another
compatible English spaCy pipeline, you can point `spacy_model_name` at that
installed model instead.

## Quick Start

```python
from labelgen import LabelGenerator, LabelGeneratorConfig

paragraphs = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
]

generator = LabelGenerator(LabelGeneratorConfig())
result = generator.fit_transform(paragraphs)

print("Concepts:")
for concept in result.concepts:
    print(concept.normalized, concept.kind, concept.document_frequency, sep=" | ")

print("Labels:")
for assignment in result.paragraph_labels:
    print(assignment.paragraph_id, assignment.label_ids, assignment.label_scores)
```

## Public API

The main public entrypoints are:

- `LabelGenerator`
- `LabelGeneratorConfig`
- `Paragraph`, `Concept`, `ConceptMention`, `Community`, `ParagraphLabels`
- `dump_result()` and `load_result()`

Detailed API notes are available in [`docs/public_api.md`](docs/public_api.md).

## Examples

Runnable examples are available in [`examples/`](examples/):

- [`examples/basic_usage.py`](examples/basic_usage.py)
- [`examples/custom_config.py`](examples/custom_config.py)
- [`examples/save_and_load.py`](examples/save_and_load.py)

## Configuration Notes

- `fit()` learns concept communities from a corpus.
- `transform()` applies previously learned communities to new paragraphs.
- `fit_transform()` learns and labels the same input in one pass.
- The default pipeline uses spaCy extraction and Leiden community detection.
- The default NLP path requires the configured spaCy model to be installed.
- `en_core_web_sm` is the recommended default model name.
- If the configured model is missing, the library raises an explicit runtime error.
- Set `use_nlp_extractor=False` to switch to the deterministic heuristic extractor.
- Set `use_graph_community_detection=False` to switch to deterministic
  connected-components community detection.
- The heuristic extractor uses capitalized spans as lightweight entities and
  non-stopword spans as candidate noun phrases.

## Opt Out Of Enhanced Implementations

```python
from labelgen import LabelGenerator, LabelGeneratorConfig

config = LabelGeneratorConfig(
    use_nlp_extractor=False,
    use_graph_community_detection=False,
)
generator = LabelGenerator(config)
```

## Use A Different spaCy Model

```python
from labelgen import LabelGenerator, LabelGeneratorConfig

config = LabelGeneratorConfig()
config.extraction.spacy_model_name = "en_core_web_md"

generator = LabelGenerator(config)
```
