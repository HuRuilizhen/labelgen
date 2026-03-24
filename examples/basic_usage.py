"""Basic end-to-end usage example for paralabelgen."""

from labelgen import LabelGenerator, LabelGeneratorConfig


def main() -> None:
    """Run a minimal fit-transform workflow with the default pipeline."""

    paragraphs = [
        "OpenAI builds language models for developers.",
        "Developers use language models in production systems.",
        "Production systems need monitoring, evaluation, and deployment tooling.",
    ]
    generator = LabelGenerator(LabelGeneratorConfig())
    result = generator.fit_transform(paragraphs)

    print("Concepts:")
    for concept in result.concepts:
        print(
            f"- {concept.normalized} | kind={concept.kind} | "
            f"document_frequency={concept.document_frequency}"
        )

    print("\nParagraph Labels:")
    for labels in result.paragraph_labels:
        print(
            f"- {labels.paragraph_id}: {labels.label_ids} | scores={labels.label_scores}"
        )


if __name__ == "__main__":
    main()
