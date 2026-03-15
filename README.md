# paralabelgen

`paralabelgen` is a Python library for generating discrete multi-label annotations for text paragraphs.

## Install

```bash
pip install paralabelgen
```

Optional extras:

```bash
pip install "paralabelgen[graph]"
pip install "paralabelgen[nlp]"
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

print("Concepts:")
for concept in result.concepts:
    print(concept.normalized, concept.kind, concept.document_frequency)

print("Labels:")
for assignment in result.paragraph_labels:
    print(assignment.paragraph_id, assignment.label_ids, assignment.label_scores)
```

## Notes

- The distribution name is `paralabelgen`, while the Python import package is `labelgen`.
- `fit` learns concept communities from a corpus.
- `transform` applies previously learned communities to new paragraphs.
- `fit_transform` learns and labels the same input in one pass.
- The base package works with deterministic fallback implementations.
- Without `paralabelgen[nlp]`, concept extraction uses regex and heuristic rules:
  capitalized spans are treated as lightweight entities, and non-stopword token spans
  are treated as candidate noun phrases.
- Without `paralabelgen[graph]`, community detection falls back to deterministic connected
  components over the concept co-occurrence graph instead of Leiden.
- Install `paralabelgen[nlp]` to enable spaCy-based concept extraction.
- Install `paralabelgen[graph]` to enable Leiden community detection.
