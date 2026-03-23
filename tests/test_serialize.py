"""Tests for result and config serialization."""

from pathlib import Path

from labelgen import LabelGenerator, LabelGeneratorConfig, dump_result, load_result


def test_result_round_trip_preserves_core_fields(tmp_path: Path) -> None:
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

    output_path = tmp_path / "result.json"
    dump_result(result, output_path)
    loaded = load_result(output_path)

    assert [paragraph.id for paragraph in loaded.paragraphs] == [
        paragraph.id for paragraph in result.paragraphs
    ]
    assert [community.display_name for community in loaded.communities] == [
        community.display_name for community in result.communities
    ]
    assert [labels.label_scores for labels in loaded.paragraph_labels] == [
        labels.label_scores for labels in result.paragraph_labels
    ]


def test_generator_save_and_load_preserve_config(tmp_path: Path) -> None:
    config = LabelGeneratorConfig()
    config.random_seed = 17
    config.use_nlp_extractor = False
    config.use_graph_community_detection = False
    config.extraction.min_document_frequency = 2
    config.label_assignment.max_labels_per_paragraph = 1

    generator = LabelGenerator(config)
    output_path = tmp_path / "generator.json"
    generator.save(output_path)
    loaded = LabelGenerator.load(output_path)

    assert loaded.config.random_seed == 17
    assert loaded.config.use_nlp_extractor is False
    assert loaded.config.use_graph_community_detection is False
    assert loaded.config.extraction.min_document_frequency == 2
    assert loaded.config.label_assignment.max_labels_per_paragraph == 1


def test_generator_save_and_load_preserve_fitted_state(tmp_path: Path) -> None:
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

    output_path = tmp_path / "fitted-generator.json"
    generator.save(output_path)
    loaded = LabelGenerator.load(output_path)
    result = loaded.transform(["OpenAI uses language models."])

    assert result.mentions
    assert result.concepts
    assert result.paragraph_labels[0].label_ids
