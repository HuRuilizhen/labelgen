"""Result and config serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from labelgen.config import (
    CommunityDetectionConfig,
    ExtractionConfig,
    ExtractorMode,
    GraphConfig,
    LabelAssignmentConfig,
    LabelGeneratorConfig,
    LLMExtractionConfig,
    VerbalizationConfig,
)
from labelgen.types import (
    Community,
    Concept,
    ConceptMention,
    GraphSummary,
    LabelGenerationResult,
    Paragraph,
    ParagraphLabels,
)


def result_to_dict(result: LabelGenerationResult) -> dict[str, Any]:
    """Convert a result object into a JSON-serializable dictionary."""

    return asdict(result)


def result_from_dict(data: dict[str, Any]) -> LabelGenerationResult:
    """Reconstruct a result object from a serialized dictionary."""

    return LabelGenerationResult(
        paragraphs=[Paragraph(**item) for item in _as_dict_list(data.get("paragraphs"))],
        concepts=[Concept(**item) for item in _as_dict_list(data.get("concepts"))],
        mentions=[ConceptMention(**item) for item in _as_dict_list(data.get("mentions"))],
        communities=[Community(**item) for item in _as_dict_list(data.get("communities"))],
        paragraph_labels=[
            ParagraphLabels(**item) for item in _as_dict_list(data.get("paragraph_labels"))
        ],
        graph_summary=_graph_summary_from_dict(data.get("graph_summary")),
        metadata=_as_string_key_dict(data.get("metadata")),
    )


def config_to_dict(config: LabelGeneratorConfig) -> dict[str, Any]:
    """Convert generator configuration into a serializable dictionary."""

    return asdict(config)


def config_from_dict(data: dict[str, Any]) -> LabelGeneratorConfig:
    """Reconstruct generator configuration from a serialized dictionary."""

    extraction_data = _as_string_key_dict(data.get("extraction"))
    llm_data = _as_string_key_dict(extraction_data.get("llm"))
    if llm_data:
        extraction_data = dict(extraction_data)
        extraction_data["llm"] = LLMExtractionConfig(**llm_data)
    extraction = ExtractionConfig(**extraction_data)
    graph = GraphConfig(**_as_string_key_dict(data.get("graph")))
    community_detection = CommunityDetectionConfig(
        **_as_string_key_dict(data.get("community_detection"))
    )
    label_assignment = LabelAssignmentConfig(
        **_as_string_key_dict(data.get("label_assignment"))
    )
    verbalization = VerbalizationConfig(**_as_string_key_dict(data.get("verbalization")))
    return LabelGeneratorConfig(
        random_seed=_as_int(data.get("random_seed"), default=42),
        extractor_mode=_as_optional_extractor_mode(data.get("extractor_mode")),
        use_nlp_extractor=_as_bool(data.get("use_nlp_extractor"), default=True),
        use_graph_community_detection=_as_bool(
            data.get("use_graph_community_detection"),
            default=True,
        ),
        extraction=extraction,
        graph=graph,
        community_detection=community_detection,
        label_assignment=label_assignment,
        verbalization=verbalization,
    )


def dump_result(result: LabelGenerationResult, path: str | Path) -> None:
    """Serialize a result object to JSON."""

    destination = Path(path)
    destination.write_text(
        json.dumps(result_to_dict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_result(path: str | Path) -> LabelGenerationResult:
    """Load a result object from JSON."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Serialized result must be a JSON object.")
    return result_from_dict(_as_string_key_dict(cast(object, data)))


def dump_config(config: LabelGeneratorConfig, path: str | Path) -> None:
    """Serialize generator configuration to JSON."""

    destination = Path(path)
    destination.write_text(
        json.dumps(config_to_dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_config(path: str | Path) -> LabelGeneratorConfig:
    """Load generator configuration from JSON."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Serialized config must be a JSON object.")
    return config_from_dict(_as_string_key_dict(cast(object, data)))


def dump_json_object(data: dict[str, Any], path: str | Path) -> None:
    """Serialize a generic JSON object to disk."""

    destination = Path(path)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a generic JSON object from disk."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    return as_json_object(cast(object, data))


def as_json_object(value: object) -> dict[str, Any]:
    """Normalize a generic object into a string-key JSON dictionary."""

    return _as_string_key_dict(value)


def as_json_object_list(value: object) -> list[dict[str, Any]]:
    """Normalize a generic object into a list of JSON dictionaries."""

    return _as_dict_list(value)


def _graph_summary_from_dict(data: object) -> GraphSummary | None:
    """Reconstruct a graph summary from serialized data."""

    if data is None:
        return None
    mapping = _as_string_key_dict(data)
    return GraphSummary(
        node_count=_as_int(mapping.get("node_count"), default=0),
        edge_count=_as_int(mapping.get("edge_count"), default=0),
        metadata=_as_string_key_dict(mapping.get("metadata")),
    )


def _as_dict_list(value: object) -> list[dict[str, Any]]:
    """Normalize a list of JSON objects."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("Expected a list of JSON objects.")

    items: list[dict[str, Any]] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise TypeError("Expected a list of JSON objects.")
        items.append(_as_string_key_dict(cast(object, item)))
    return items


def _as_string_key_dict(value: object) -> dict[str, Any]:
    """Normalize a JSON object into a string-key dictionary."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object.")

    normalized: dict[str, Any] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError("JSON object keys must be strings.")
        normalized[key] = item
    return normalized


def _as_int(value: object, *, default: int) -> int:
    """Normalize integer values from serialized data."""

    if value is None:
        return default
    if not isinstance(value, int):
        raise TypeError("Expected an integer value.")
    return value


def _as_bool(value: object, *, default: bool) -> bool:
    """Normalize boolean values from serialized data."""

    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError("Expected a boolean value.")
    return value


def _as_optional_extractor_mode(value: object) -> ExtractorMode | None:
    """Normalize optional extractor mode values from serialized data."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Expected a string value.")
    if value not in {"spacy", "heuristic", "llm"}:
        raise TypeError("Unsupported extractor mode in serialized config.")
    return cast(ExtractorMode, value)
