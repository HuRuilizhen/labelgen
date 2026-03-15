"""Smoke tests for the main pipeline."""

from labelgen import LabelGenerator, LabelGeneratorConfig


def test_fit_transform_returns_structured_result() -> None:
    generator = LabelGenerator(LabelGeneratorConfig())
    result = generator.fit_transform(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    assert len(result.paragraphs) == 2
    assert result.mentions
    assert result.concepts
    assert result.graph_summary is not None
    assert result.graph_summary.node_count > 0


def test_min_document_frequency_filters_single_paragraph_concepts() -> None:
    config = LabelGeneratorConfig()
    config.extraction.min_document_frequency = 2

    generator = LabelGenerator(config)
    result = generator.fit_transform(
        [
            "OpenAI builds language models.",
            "OpenAI deploys systems.",
        ]
    )

    assert result.concepts
    assert all((concept.document_frequency or 0) >= 2 for concept in result.concepts)
