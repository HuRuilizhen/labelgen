# labelgen

`labelgen` is a Python library for generating discrete multi-label annotations for text paragraphs.

## Install

```bash
pip install labelgen
```

Optional extras:

```bash
pip install "labelgen[graph]"
pip install "labelgen[nlp]"
```

## Example

```python
from labelgen import LabelGenerator, LabelGeneratorConfig

paragraphs = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
]

generator = LabelGenerator(LabelGeneratorConfig())
result = generator.fit_transform(paragraphs)

for assignment in result.paragraph_labels:
    print(assignment.paragraph_id, assignment.label_ids, assignment.label_scores)
```

## Notes

- `fit` learns concept communities from a corpus.
- `transform` applies previously learned communities to new paragraphs.
- `fit_transform` learns and labels the same input in one pass.
- The base package works with deterministic fallback implementations.
- Install `labelgen[graph]` to enable Leiden community detection.
- Install `labelgen[nlp]` to enable spaCy-based concept extraction.
