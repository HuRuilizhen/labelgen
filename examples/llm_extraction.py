"""LLM-backed concept extraction example.

This example uses the v0.2.0 LLM extractor path. Configure the provider and
model through environment variables before running it.

Example:

    export LABELGEN_LLM_PROVIDER=openai
    export LABELGEN_LLM_MODEL=gpt-5-mini
    export OPENAI_API_KEY=...
    python examples/llm_extraction.py
"""

from __future__ import annotations

import os

from labelgen import LabelGenerator, LabelGeneratorConfig


def main() -> None:
    """Run a small LLM extraction workflow and print concepts and labels."""

    provider = os.environ.get("LABELGEN_LLM_PROVIDER", "openai")
    model = os.environ.get("LABELGEN_LLM_MODEL", "")
    base_url = os.environ.get("LABELGEN_LLM_BASE_URL")
    api_key_env_var = os.environ.get("LABELGEN_LLM_API_KEY_ENV_VAR")

    if not model:
        raise RuntimeError(
            "Set LABELGEN_LLM_MODEL before running the LLM extraction example."
        )

    config = LabelGeneratorConfig(
        extractor_mode="llm",
        use_graph_community_detection=False,
    )
    config.extraction.llm.provider = provider  # type: ignore[assignment]
    config.extraction.llm.model = model
    config.extraction.llm.base_url = base_url
    config.extraction.llm.api_key_env_var = api_key_env_var
    config.extraction.llm.cache_enabled = True
    config.extraction.llm.cache_dir = ".labelgen-cache"
    config.extraction.llm.batch_size = 2
    config.extraction.llm.max_concepts_per_paragraph = 8

    paragraphs = [
        "OpenAI builds language models and developer APIs for production systems.",
        "Mistral provides foundation models and inference endpoints for enterprise use.",
        "Qwen models are commonly used for multilingual generation and reasoning tasks.",
    ]

    generator = LabelGenerator(config)
    result = generator.fit_transform(paragraphs)

    print(f"Extractor: {generator.extractor_name}")
    print(f"Detector: {generator.detector_name}")
    print("Concepts:")
    for concept in result.concepts:
        print(
            f"- {concept.normalized} | kind={concept.kind} | "
            f"document_frequency={concept.document_frequency}"
        )

    print("\nParagraph Labels:")
    for labels in result.paragraph_labels:
        print(f"- {labels.paragraph_id}: {labels.label_ids} | scores={labels.label_scores}")


if __name__ == "__main__":
    main()
