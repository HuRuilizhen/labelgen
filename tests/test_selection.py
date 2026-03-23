"""Tests for component selection and default behavior."""

from labelgen import LabelGenerator, LabelGeneratorConfig


def test_generator_uses_enhanced_components_by_default() -> None:
    generator = LabelGenerator(LabelGeneratorConfig())

    assert generator.extractor_name == "SpacyConceptExtractor"
    assert generator.detector_name == "LeidenCommunityDetector"


def test_generator_supports_explicit_opt_out_configuration() -> None:
    config = LabelGeneratorConfig(
        use_nlp_extractor=False,
        use_graph_community_detection=False,
    )

    generator = LabelGenerator(config)

    assert generator.extractor_name == "HeuristicConceptExtractor"
    assert generator.detector_name == "ConnectedComponentsCommunityDetector"
