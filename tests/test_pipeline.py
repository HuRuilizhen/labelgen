"""Smoke tests for the main pipeline."""

import pytest

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


def test_transform_requires_fit() -> None:
    generator = LabelGenerator(LabelGeneratorConfig())

    with pytest.raises(RuntimeError):
        generator.transform(["OpenAI builds language models."])


def test_transform_uses_fitted_communities() -> None:
    generator = LabelGenerator(LabelGeneratorConfig())
    generator.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers deploy language models in products.",
        ]
    )

    result = generator.transform(["Developers use language models."])

    assert result.communities
    assert result.paragraph_labels[0].label_ids
    assert result.metadata["is_fitted"] is True


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


def test_max_concept_df_ratio_filters_corpus_wide_concepts() -> None:
    config = LabelGeneratorConfig()
    config.extraction.max_concept_df_ratio = 0.5

    generator = LabelGenerator(config)
    result = generator.fit_transform(
        [
            "OpenAI builds systems.",
            "OpenAI deploys models.",
            "OpenAI serves APIs.",
        ]
    )

    assert all((concept.document_frequency or 0) / 3 <= 0.5 for concept in result.concepts)
