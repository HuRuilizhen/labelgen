# paralabelgen

`paralabelgen` is a Python library for generating discrete paragraph labels
from concept extraction, graph communities, and interpretable assignment rules.

- PyPI distribution: `paralabelgen`
- Python import package: `labelgen`
- Repository: `https://github.com/HuRuilizhen/labelgen`

## Install

```bash
pip install paralabelgen
```

If you want to use the default spaCy extractor, install a compatible English
pipeline such as:

```bash
python -m spacy download en_core_web_sm
```

`en_core_web_sm` is the recommended default model, but you can point
`spacy_model_name` at another installed compatible spaCy pipeline.

## Quick Start

### Default spaCy pipeline

```python
from labelgen import LabelGenerator, LabelGeneratorConfig

paragraphs = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
]

generator = LabelGenerator(LabelGeneratorConfig())
result = generator.fit_transform(paragraphs)

for concept in result.concepts:
    print(concept.normalized, concept.kind, concept.document_frequency, sep=" | ")

for assignment in result.paragraph_labels:
    print(assignment.paragraph_id, assignment.label_ids, assignment.label_scores)
```

### LLM extraction pipeline

```python
from labelgen import LabelGenerator, LabelGeneratorConfig

config = LabelGeneratorConfig(
    extractor_mode="llm",
    use_graph_community_detection=False,
)
config.extraction.llm.provider = "openai"
config.extraction.llm.model = "gpt-5-mini"

generator = LabelGenerator(config)
result = generator.fit_transform(
    [
        "OpenAI builds language models and developer APIs for production systems.",
        "Production systems need monitoring and evaluation tooling.",
    ]
)
```

The LLM extractor supports `openai`, `mistral`, and `qwen` style providers.
Set the corresponding API key in the expected environment variable:

- `OPENAI_API_KEY`
- `MISTRAL_API_KEY`
- `DASHSCOPE_API_KEY`

## Extraction Modes

`LabelGeneratorConfig.extractor_mode` supports three modes:

- `spacy`: default public extractor using spaCy noun chunks and entities
- `heuristic`: deterministic fallback extractor using rule-based spans
- `llm`: provider-backed concept extraction using structured JSON output

If `extractor_mode` is unset, the legacy `use_nlp_extractor` compatibility flag
is still respected. New code should prefer `extractor_mode`.

## LLM Configuration Notes

The LLM extraction path is opt-in and synchronous. Key settings live under
`config.extraction.llm`:

- `provider`
- `model`
- `api_key_env_var`
- `base_url`
- `temperature`
- `max_output_tokens`
- `batch_size`
- `max_concepts_per_paragraph`
- `cache_enabled`
- `cache_dir`
- `record_extraction_artifacts`
- `artifact_dir`
- `prompt_version`
- `prompt_template`

Cache and artifact behavior:

- `cache_enabled=True` stores parsed concept lists on disk and avoids repeated
  provider calls for the same effective request
- `record_extraction_artifacts=True` writes structured per-batch extraction
  artifacts for audit and experiment analysis
- both are optional and can be disabled independently

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
- [`examples/llm_extraction.py`](examples/llm_extraction.py)

## Configuration Notes

- `fit()` learns concepts and communities from a corpus
- `transform()` applies previously learned communities to new paragraphs
- `fit_transform()` learns and labels the same input in one pass
- `use_graph_community_detection=True` uses Leiden community detection
- `use_graph_community_detection=False` uses deterministic connected components
- the default spaCy path requires the configured spaCy model to be installed
- the LLM path does not silently fall back to spaCy or heuristic extraction

