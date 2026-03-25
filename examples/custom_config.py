"""Custom configuration example for paralabelgen."""

from labelgen import LabelGenerator, LabelGeneratorConfig


def main() -> None:
    """Run the pipeline with explicit deterministic fallback settings."""

    config = LabelGeneratorConfig(
        extractor_mode="heuristic",
        use_graph_community_detection=False,
    )
    config.extraction.min_document_frequency = 2
    config.label_assignment.max_labels_per_paragraph = 2

    paragraphs = [
        "OpenAI builds language models for developers.",
        "Developers use language models in production systems.",
        "Production systems rely on deployment tooling and evaluation.",
    ]

    generator = LabelGenerator(config)
    result = generator.fit_transform(paragraphs)

    print(f"Extractor: {generator.extractor_name}")
    print(f"Detector: {generator.detector_name}")
    print("Communities:")
    for community in result.communities:
        print(f"- {community.id}: {community.display_name}")


if __name__ == "__main__":
    main()
