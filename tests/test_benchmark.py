"""Tests for local benchmark helper scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from labelgen import LabelGenerator, LabelGeneratorConfig


def _load_module(path: str, name: str) -> ModuleType:
    """Load a local benchmark helper module by file path."""

    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_benchmark = _load_module("benchmark/run_benchmark.py", "benchmark_run_benchmark")
summarize_results_module = _load_module(
    "benchmark/summarize_results.py",
    "benchmark_summarize_results",
)


def test_load_records_supports_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "text": "OpenAI builds APIs."}),
                json.dumps({"text": "Mistral ships models."}),
            ]
        ),
        encoding="utf-8",
    )

    records = run_benchmark.load_records(input_path)

    assert records == [
        {"id": "p1", "text": "OpenAI builds APIs."},
        {"text": "Mistral ships models."},
    ]


def test_load_records_supports_json_records_object(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.json"
    input_path.write_text(
        json.dumps(
            {
                "records": [
                    {"id": "p1", "text": "OpenAI builds APIs."},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = run_benchmark.load_records(input_path)

    assert records == [{"id": "p1", "text": "OpenAI builds APIs."}]


def test_build_paragraphs_requires_text_and_allows_optional_id() -> None:
    paragraphs = run_benchmark.build_paragraphs(
        [
            {"id": "given-id", "text": "OpenAI builds APIs."},
            {"text": "Mistral ships models.", "source": "tech-note"},
        ]
    )

    assert paragraphs[0].id == "given-id"
    assert paragraphs[1].id == "benchmark-1"
    assert paragraphs[1].metadata == {"source": "tech-note", "benchmark_index": 1}


def test_summarize_run_reports_basic_benchmark_fields() -> None:
    config = LabelGeneratorConfig(
        use_nlp_extractor=False,
        use_graph_community_detection=False,
    )
    generator = LabelGenerator(config)
    paragraphs = run_benchmark.build_paragraphs(
        [
            {"id": "p1", "text": "OpenAI builds language models."},
            {"id": "p2", "text": "Mistral provides inference endpoints."},
        ]
    )
    result = generator.fit_transform(paragraphs)

    args = argparse.Namespace(
        extractor="heuristic",
        provider=None,
        model=None,
        output_contract_mode="auto",
        sample_preview=2,
        batch_size=8,
        max_output_tokens=512,
        timeout_seconds=30.0,
        max_concepts_per_paragraph=12,
        record_artifacts=False,
        artifact_dir=None,
        record_raw_response_text=False,
        record_paragraph_text=False,
        record_paragraph_metadata=False,
    )
    summary = run_benchmark.summarize_run(
        args=args,
        paragraphs=paragraphs,
        result=result,
    )

    assert summary["paragraph_count"] == 2
    assert summary["concept_count"] > 0
    assert summary["mention_count"] > 0
    assert len(summary["preview"]) == 2


def test_summarize_run_builds_preview_from_cleaned_result_paragraphs() -> None:
    config = LabelGeneratorConfig(
        use_nlp_extractor=False,
        use_graph_community_detection=False,
    )
    generator = LabelGenerator(config)
    paragraphs = run_benchmark.build_paragraphs(
        [
            {"id": "url-only", "text": "https://example.com/support"},
            {"id": "kept", "text": "OpenAI builds language models."},
        ]
    )
    result = generator.fit_transform(paragraphs)

    args = argparse.Namespace(
        extractor="heuristic",
        provider=None,
        model=None,
        output_contract_mode="auto",
        sample_preview=2,
        batch_size=8,
        max_output_tokens=512,
        timeout_seconds=30.0,
        max_concepts_per_paragraph=12,
        record_artifacts=False,
        artifact_dir=None,
        record_raw_response_text=False,
        record_paragraph_text=False,
        record_paragraph_metadata=False,
    )
    summary = run_benchmark.summarize_run(
        args=args,
        paragraphs=paragraphs,
        result=result,
    )

    assert summary["paragraph_count"] == 2
    assert len(result.paragraphs) == 1
    assert len(summary["preview"]) == 1
    assert summary["preview"][0]["source_id"] == "kept"
    assert summary["preview"][0]["text"] == "OpenAI builds language models"


def test_summarize_results_and_markdown_render() -> None:
    summary = summarize_results_module.summarize_results(
        [
            {
                "extractor": "heuristic",
                "provider": None,
                "model": None,
                "output_contract_mode": None,
                "paragraph_count": 10,
                "concept_count": 20,
                "mention_count": 30,
                "empty_paragraph_count": 1,
                "average_concepts_per_paragraph": 2.0,
                "average_mentions_per_paragraph": 3.0,
            },
            {
                "extractor": "llm",
                "provider": "ollama",
                "model": "qwen3.5:4b",
                "output_contract_mode": "auto",
                "paragraph_count": 10,
                "concept_count": 12,
                "mention_count": 12,
                "empty_paragraph_count": 0,
                "average_concepts_per_paragraph": 1.2,
                "average_mentions_per_paragraph": 1.2,
            },
        ]
    )

    markdown = summarize_results_module.render_markdown(summary)

    assert summary["runs"][0]["run"] == "heuristic"
    assert summary["runs"][1]["run"] == "llm:ollama:qwen3.5:4b"
    assert "| heuristic | 10 | 20 | 30 | 1 | 2.00 | 3.00 |" in markdown


def test_build_config_uses_conservative_default_batch_size_for_ollama() -> None:
    args = argparse.Namespace(
        extractor="llm",
        provider="ollama",
        model="qwen3.5:4b",
        output_contract_mode="auto",
        sample_preview=2,
        batch_size=None,
        max_output_tokens=512,
        timeout_seconds=30.0,
        max_concepts_per_paragraph=12,
        record_artifacts=False,
        artifact_dir=None,
        record_raw_response_text=False,
        record_paragraph_text=False,
        record_paragraph_metadata=False,
    )

    config = run_benchmark.build_config(args)

    assert config.extraction.llm.batch_size == 1


def test_build_config_keeps_default_cloud_batch_size() -> None:
    args = argparse.Namespace(
        extractor="llm",
        provider="mistral",
        model="mistral-small",
        output_contract_mode="auto",
        sample_preview=2,
        batch_size=None,
        max_output_tokens=512,
        timeout_seconds=30.0,
        max_concepts_per_paragraph=12,
        record_artifacts=False,
        artifact_dir=None,
        record_raw_response_text=False,
        record_paragraph_text=False,
        record_paragraph_metadata=False,
    )

    config = run_benchmark.build_config(args)

    assert config.extraction.llm.batch_size == 8


def test_build_config_can_enable_benchmark_artifacts() -> None:
    args = argparse.Namespace(
        extractor="llm",
        provider="ollama",
        model="qwen3.5:4b",
        output_contract_mode="auto",
        sample_preview=2,
        batch_size=None,
        max_output_tokens=512,
        timeout_seconds=30.0,
        max_concepts_per_paragraph=12,
        record_artifacts=True,
        artifact_dir="experiment/artifacts/benchmark-ollama",
        record_raw_response_text=True,
        record_paragraph_text=True,
        record_paragraph_metadata=True,
    )

    config = run_benchmark.build_config(args)

    assert config.extraction.llm.record_extraction_artifacts is True
    assert config.extraction.llm.artifact_dir == "experiment/artifacts/benchmark-ollama"
    assert config.extraction.llm.record_raw_response_text is True
    assert config.extraction.llm.record_paragraph_text is True
    assert config.extraction.llm.record_paragraph_metadata is True
