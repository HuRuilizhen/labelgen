"""Summarize multiple local benchmark result files into one compact comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark summarization."""

    parser = argparse.ArgumentParser(
        description="Summarize multiple paralabelgen benchmark result files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more benchmark result JSON files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON or Markdown file.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Summary output format.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    """Load one benchmark result summary from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Benchmark result files must contain one JSON object.")
    return cast(dict[str, Any], payload)


def run_label(result: dict[str, Any]) -> str:
    """Return a compact label for one benchmark run."""

    extractor = result.get("extractor")
    provider = result.get("provider")
    model = result.get("model")
    if extractor != "llm":
        return str(extractor)
    parts = [str(extractor)]
    if provider:
        parts.append(str(provider))
    if model:
        parts.append(str(model))
    return ":".join(parts)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a machine-readable summary across benchmark runs."""

    runs: list[dict[str, Any]] = []
    for result in results:
        runs.append(
            {
                "run": run_label(result),
                "extractor": result.get("extractor"),
                "provider": result.get("provider"),
                "model": result.get("model"),
                "output_contract_mode": result.get("output_contract_mode"),
                "paragraph_count": result.get("paragraph_count"),
                "concept_count": result.get("concept_count"),
                "mention_count": result.get("mention_count"),
                "empty_paragraph_count": result.get("empty_paragraph_count"),
                "average_concepts_per_paragraph": result.get("average_concepts_per_paragraph"),
                "average_mentions_per_paragraph": result.get("average_mentions_per_paragraph"),
            }
        )
    return {"runs": runs}


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a Markdown comparison table from a summary payload."""

    lines = [
        "| Run | Paragraphs | Concepts | Mentions | Empty | Avg Concepts | Avg Mentions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        lines.append(
            (
                "| {run} | {paragraphs} | {concepts} | {mentions} | {empty} | "
                "{avg_concepts:.2f} | {avg_mentions:.2f} |"
            ).format(
                run=run["run"],
                paragraphs=run["paragraph_count"],
                concepts=run["concept_count"],
                mentions=run["mention_count"],
                empty=run["empty_paragraph_count"],
                avg_concepts=run["average_concepts_per_paragraph"],
                avg_mentions=run["average_mentions_per_paragraph"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read benchmark results and write a compact summary."""

    args = parse_args()
    results = [load_result(Path(path)) for path in args.inputs]
    summary = summarize_results(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    output_path.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
