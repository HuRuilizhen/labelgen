# paralabelgen

`paralabelgen` is a Python library for generating discrete multi-label annotations for text paragraphs.

## Install

```bash
pip install paralabelgen
python -m spacy download en_core_web_sm
```

`en_core_web_sm` is the recommended default model. If you already use another
compatible English spaCy pipeline, you can point `spacy_model_name` at that
installed model instead.

## Example

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
    print(concept.normalized, concept.kind, concept.document_frequency)

print("Labels:")
for assignment in result.paragraph_labels:
    print(assignment.paragraph_id, assignment.label_ids, assignment.label_scores)
```

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

## Notes

- The distribution name is `paralabelgen`, while the Python import package is `labelgen`.
- `fit` learns concept communities from a corpus.
- `transform` applies previously learned communities to new paragraphs.
- `fit_transform` learns and labels the same input in one pass.
- `0.1.0` installs spaCy extraction and Leiden community detection by default.
- The default NLP path requires the configured spaCy model to be installed.
- `en_core_web_sm` is the recommended default model name.
- If the model is missing, the library raises an explicit runtime error instead of silently
  falling back.
- Set `use_nlp_extractor=False` to switch to the deterministic heuristic extractor.
- Set `use_graph_community_detection=False` to switch to deterministic connected-components
  community detection.
- The heuristic extractor uses capitalized spans as lightweight entities and non-stopword
  spans as candidate noun phrases.
