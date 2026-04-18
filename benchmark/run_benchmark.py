"""Run local extractor benchmarks against a JSON or JSONL paragraph dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict, cast

from labelgen import LabelGenerator, LabelGeneratorConfig
from labelgen.types import Paragraph


class BenchmarkRecord(TypedDict, total=False):
    """One benchmark input record loaded from JSON or JSONL."""

    id: str
    text: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one benchmark run."""

    parser = argparse.ArgumentParser(
        description="Run paralabelgen benchmark comparisons on a local dataset."
    )
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL dataset file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON summary file.",
    )
    parser.add_argument(
        "--extractor",
        choices=("heuristic", "spacy", "llm"),
        required=True,
        help="Extractor mode to benchmark.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "mistral", "qwen", "ollama", "deepseek"),
        help="LLM provider when --extractor=llm.",
    )
    parser.add_argument("--model", help="LLM model name when --extractor=llm.")
    parser.add_argument(
        "--output-contract-mode",
        choices=("auto", "json_schema", "json_object", "prompt_only"),
        default="auto",
        help="Output contract mode when --extractor=llm.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "LLM batch size when --extractor=llm. "
            "Defaults to 8 for cloud providers and 1 for Ollama."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="LLM max output tokens when --extractor=llm.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="LLM timeout when --extractor=llm.",
    )
    parser.add_argument(
        "--max-concepts-per-paragraph",
        type=int,
        default=12,
        help="LLM cap for parsed concepts per paragraph.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional maximum number of input paragraphs to benchmark.",
    )
    parser.add_argument(
        "--sample-preview",
        type=int,
        default=5,
        help="Number of sample paragraphs to retain in the output preview.",
    )
    parser.add_argument(
        "--record-artifacts",
        action="store_true",
        help="Record LLM extraction artifacts for debugging benchmark failures.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Artifact output directory when --record-artifacts is enabled.",
    )
    parser.add_argument(
        "--record-raw-response-text",
        action="store_true",
        help="Include raw provider response text in recorded benchmark artifacts.",
    )
    parser.add_argument(
        "--record-paragraph-text",
        action="store_true",
        help="Include paragraph text in recorded benchmark artifacts.",
    )
    parser.add_argument(
        "--record-paragraph-metadata",
        action="store_true",
        help="Include paragraph metadata in recorded benchmark artifacts.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load benchmark input records from a JSON or JSONL file."""

    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise RuntimeError("Each JSONL benchmark record must be a JSON object.")
            records.append(cast(dict[str, Any], item))
        return records

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_records: list[object]
        if isinstance(payload, list):
            raw_records = cast(list[object], payload)
        elif isinstance(payload, dict):
            payload_dict = cast(dict[str, Any], payload)
            records_value = payload_dict.get("records")
            if not isinstance(records_value, list):
                raise RuntimeError(
                    "JSON benchmark input must be a list or an object with `records`."
                )
            raw_records = cast(list[object], records_value)
        else:
            raise RuntimeError("JSON benchmark input must be a list or an object with `records`.")
        normalized: list[dict[str, Any]] = []
        for item in raw_records:
            if not isinstance(item, dict):
                raise RuntimeError("Each JSON benchmark record must be a JSON object.")
            normalized.append(cast(dict[str, Any], item))
        return normalized

    raise RuntimeError("Benchmark input must be a .json or .jsonl file.")


def build_paragraphs(records: list[dict[str, Any]]) -> list[Paragraph]:
    """Convert benchmark input records into paragraph models."""

    paragraphs: list[Paragraph] = []
    for index, record in enumerate(records):
        normalized_record = cast(BenchmarkRecord, record)
        text = normalized_record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Each benchmark record must contain a non-empty `text` field.")
        sample_id = normalized_record.get("id")
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"id", "text"}
        }
        paragraphs.append(
            Paragraph(
                id=sample_id if isinstance(sample_id, str) and sample_id else f"benchmark-{index}",
                text=text,
                metadata=metadata or None,
            )
        )
        if not isinstance(sample_id, str) or not sample_id:
            # Preserve a stable external sample identifier in metadata when the
            # caller did not provide one explicitly.
            paragraphs[-1].metadata = {
                **(paragraphs[-1].metadata or {}),
                "benchmark_index": index,
            }
    return paragraphs


def build_config(args: argparse.Namespace) -> LabelGeneratorConfig:
    """Build one benchmark configuration from CLI arguments."""

    config = LabelGeneratorConfig(
        extractor_mode=args.extractor,
        use_graph_community_detection=False,
    )
    if args.extractor == "llm":
        if not args.provider or not args.model:
            raise RuntimeError("--provider and --model are required when --extractor=llm.")
        config.extraction.llm.provider = args.provider
        config.extraction.llm.model = args.model
        config.extraction.llm.output_contract_mode = args.output_contract_mode
        config.extraction.llm.batch_size = _resolve_llm_batch_size(args)
        config.extraction.llm.max_output_tokens = args.max_output_tokens
        config.extraction.llm.timeout_seconds = args.timeout_seconds
        config.extraction.llm.max_concepts_per_paragraph = args.max_concepts_per_paragraph
        config.extraction.llm.record_extraction_artifacts = args.record_artifacts
        if args.artifact_dir is not None:
            config.extraction.llm.artifact_dir = args.artifact_dir
        config.extraction.llm.record_raw_response_text = args.record_raw_response_text
        config.extraction.llm.record_paragraph_text = args.record_paragraph_text
        config.extraction.llm.record_paragraph_metadata = args.record_paragraph_metadata
    return config


def _resolve_llm_batch_size(args: argparse.Namespace) -> int:
    """Resolve a provider-aware benchmark batch size for one run."""

    if args.batch_size is not None:
        return args.batch_size
    if args.provider == "ollama":
        return 1
    return 8


def build_preview(
    result: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Build a compact sample preview for the benchmark output."""

    concept_by_id = {concept.id: concept.normalized for concept in result.concepts}
    mention_map: dict[str, list[str]] = {}
    for mention in result.mentions:
        mention_map.setdefault(mention.paragraph_id, []).append(mention.normalized)

    preview: list[dict[str, Any]] = []
    for paragraph, labels in zip(
        result.paragraphs[:limit],
        result.paragraph_labels[:limit],
        strict=True,
    ):
        preview.append(
            {
                "paragraph_id": labels.paragraph_id,
                "source_id": paragraph.id,
                "text": paragraph.text,
                "concepts": mention_map.get(labels.paragraph_id, []),
                "labels": labels.label_ids,
                "label_scores": labels.label_scores,
                "label_concepts": {
                    label_id: [
                        concept_by_id.get(concept_id, concept_id)
                        for concept_id in labels.evidence_concept_ids
                    ]
                    for label_id in labels.label_ids
                },
            }
        )
    return preview


def summarize_run(
    *,
    args: argparse.Namespace,
    paragraphs: list[Paragraph],
    result: Any,
) -> dict[str, Any]:
    """Build the machine-readable benchmark summary for one run."""

    paragraph_count = len(paragraphs)
    concept_count = len(result.concepts)
    mention_count = len(result.mentions)
    paragraphs_with_mentions = {mention.paragraph_id for mention in result.mentions}
    empty_paragraph_count = sum(
        1
        for labels in result.paragraph_labels
        if labels.paragraph_id not in paragraphs_with_mentions
    )

    return {
        "extractor": args.extractor,
        "provider": args.provider,
        "model": args.model,
        "output_contract_mode": args.output_contract_mode if args.extractor == "llm" else None,
        "paragraph_count": paragraph_count,
        "concept_count": concept_count,
        "mention_count": mention_count,
        "empty_paragraph_count": empty_paragraph_count,
        "average_concepts_per_paragraph": (
            concept_count / paragraph_count if paragraph_count else 0.0
        ),
        "average_mentions_per_paragraph": (
            mention_count / paragraph_count if paragraph_count else 0.0
        ),
        "preview": build_preview(result, limit=args.sample_preview),
        "config": asdict(build_config(args)),
    }


def main() -> None:
    """Run one benchmark configuration and write a JSON summary."""

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_records(input_path)
    if args.sample_limit is not None:
        records = records[: args.sample_limit]
    paragraphs = build_paragraphs(records)

    generator = LabelGenerator(build_config(args))
    result = generator.fit_transform(paragraphs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            summarize_run(
                args=args,
                paragraphs=paragraphs,
                result=result,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
