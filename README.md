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

## Extraction Modes

`LabelGeneratorConfig.extractor_mode` supports three modes:

- `spacy`: default public extractor using spaCy noun chunks and entities
- `heuristic`: deterministic fallback extractor using rule-based spans
- `llm`: provider-backed concept extraction using a unified OpenAI-compatible
  chat-completions client

If `extractor_mode` is unset, the legacy `use_nlp_extractor` compatibility flag
is still respected. New code should prefer `extractor_mode`.

## LLM Provider Model

The LLM extraction path is opt-in and synchronous. The current provider layer is
unified around one OpenAI-compatible client and supports:

- `openai`
- `mistral`
- `qwen`

Configure the provider and model under `config.extraction.llm`:

- `provider`
- `model`
- `api_key_env_var`
- `base_url`
- `organization`
- `timeout_seconds`
- `max_retries`
- `temperature`
- `max_output_tokens`
- `batch_size`
- `max_concepts_per_paragraph`

Set the corresponding API key in the expected environment variable:

- `OPENAI_API_KEY`
- `MISTRAL_API_KEY`
- `DASHSCOPE_API_KEY`

## Structured Output And Reliability

The LLM extractor now prefers provider-enforced structured output when the
configured endpoint supports OpenAI-compatible JSON schema response formatting.

- prompt guidance is still used, but it is no longer the only output contract
- structured output is enforced first when available
- if an OpenAI-compatible endpoint rejects JSON-schema response formatting, the
  client falls back to the prompt-only request on the same LLM path
- the extractor does not silently fall back to `spacy` or `heuristic`

## Recommended LLM Settings

### Low-risk evaluation workflow

For routine evaluation runs, prefer a conservative configuration:

- `temperature = 0.0`
- `batch_size = 1` or a small batch size
- `cache_enabled = True`
- `record_extraction_artifacts = False`

This keeps runs reproducible and avoids writing extra local artifacts unless you
actually need them.

### Debugging-oriented workflow

When you need to inspect provider behavior, you can enable artifacts:

- `record_extraction_artifacts = True`
- `record_raw_response_text = True` only when raw provider output is needed
- `record_paragraph_text = True` only when paragraph text is safe to store
- `record_paragraph_metadata = True` only when metadata is safe to store

Artifact recording is optional and should stay disabled by default for routine
usage.

## Cache And Artifact Notes

- `cache_enabled=True` stores parsed concept lists on disk and avoids repeated
  provider calls for the same effective request
- cache invalidation includes both `prompt_version` and the effective prompt
  text
- artifacts are intended for local evaluation and debugging workflows, not as a
  default production feature

## Optional Manual Smoke Test

For a small manual LLM-path verification outside the default test suite, run
the example script with one provider/model pair and a valid API key in the
expected environment variable:

```bash
OPENAI_API_KEY=... .venv/bin/python examples/llm_extraction.py
```

This is intended as a lightweight manual smoke test for provider connectivity
and parsing, not as part of the default automated suite.

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
- the LLM path requires valid provider configuration and credentials
