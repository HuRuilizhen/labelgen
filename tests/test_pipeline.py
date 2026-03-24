"""Smoke tests for the main pipeline."""

from types import SimpleNamespace

import pytest

from labelgen import LabelGenerator, LabelGeneratorConfig
from labelgen.extraction.spacy_extractor import SpacyConceptExtractor


def test_fit_transform_returns_structured_result() -> None:
    generator = LabelGenerator(
        LabelGeneratorConfig(
            use_nlp_extractor=False,
            use_graph_community_detection=False,
        )
    )
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
    generator = LabelGenerator(
        LabelGeneratorConfig(
            use_nlp_extractor=False,
            use_graph_community_detection=False,
        )
    )

    with pytest.raises(RuntimeError):
        generator.transform(["OpenAI builds language models."])


def test_transform_uses_fitted_communities() -> None:
    generator = LabelGenerator(
        LabelGeneratorConfig(
            use_nlp_extractor=False,
            use_graph_community_detection=False,
        )
    )
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


def test_transform_does_not_reapply_training_df_threshold_to_inference_input() -> None:
    config = LabelGeneratorConfig(
        use_nlp_extractor=False,
        use_graph_community_detection=False,
    )
    config.extraction.min_document_frequency = 2

    generator = LabelGenerator(config)
    generator.fit(
        [
            "OpenAI builds language models.",
            "OpenAI deploys language models.",
        ]
    )

    result = generator.transform(["OpenAI uses language models."])

    assert result.mentions
    assert result.concepts
    assert result.paragraph_labels[0].label_ids


def test_min_document_frequency_filters_single_paragraph_concepts() -> None:
    config = LabelGeneratorConfig(
        use_nlp_extractor=False,
        use_graph_community_detection=False,
    )
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
    config = LabelGeneratorConfig(
        use_nlp_extractor=False,
        use_graph_community_detection=False,
    )
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


def test_default_nlp_extractor_requires_installed_spacy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_import_module(name: str) -> object:
        assert name == "spacy"
        return SimpleNamespace(load=_fake_load)

    def _fake_load(name: str) -> object:
        raise OSError(f"missing model: {name}")

    monkeypatch.setattr(
        "labelgen.extraction.spacy_extractor.importlib.import_module",
        _fake_import_module,
    )
    generator = LabelGenerator(LabelGeneratorConfig())

    with pytest.raises(RuntimeError, match="spaCy model 'en_core_web_sm' is required"):
        generator.fit_transform(["OpenAI builds language models."])


def test_default_nlp_extractor_wraps_model_load_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_import_module(name: str) -> object:
        assert name == "spacy"
        return SimpleNamespace(load=_fake_load)

    def _fake_load(name: str) -> object:
        raise ValueError(f"broken model: {name}")

    monkeypatch.setattr(
        "labelgen.extraction.spacy_extractor.importlib.import_module",
        _fake_import_module,
    )
    extractor = SpacyConceptExtractor(LabelGeneratorConfig().extraction)

    with pytest.raises(RuntimeError, match="could not be loaded"):
        extractor.extract([])
