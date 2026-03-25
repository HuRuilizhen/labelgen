"""Generator persistence example for paralabelgen."""

from pathlib import Path

from labelgen import LabelGenerator, LabelGeneratorConfig


def main() -> None:
    """Fit, save, load, and reuse a generator."""

    config = LabelGeneratorConfig(
        extractor_mode="heuristic",
        use_graph_community_detection=False,
    )
    generator = LabelGenerator(config)
    generator.fit(
        [
            "OpenAI builds language models.",
            "OpenAI deploys language models.",
        ]
    )

    output_path = Path("generator-example.json")
    generator.save(output_path)

    loaded = LabelGenerator.load(output_path)
    result = loaded.transform(["OpenAI uses language models in production."])

    print(f"Saved generator to: {output_path}")
    print(f"Labels: {result.paragraph_labels[0].label_ids}")
    output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
