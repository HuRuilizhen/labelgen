"""Tests for result and config serialization."""

import json
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
    config.extractor_mode = "llm"
    config.use_nlp_extractor = False
    config.use_graph_community_detection = False
    config.extraction.min_document_frequency = 2
    config.extraction.llm.provider = "mistral"
    config.extraction.llm.model = "mistral-small-latest"
    config.label_assignment.max_labels_per_paragraph = 1

    generator = LabelGenerator(config)
    output_path = tmp_path / "generator.json"
    generator.save(output_path)
    loaded = LabelGenerator.load(output_path)

    assert loaded.config.random_seed == 17
    assert loaded.config.extractor_mode == "llm"
    assert loaded.config.use_nlp_extractor is False
    assert loaded.config.use_graph_community_detection is False
    assert loaded.config.extraction.min_document_frequency == 2
    assert loaded.config.extraction.llm.provider == "mistral"
    assert loaded.config.extraction.llm.model == "mistral-small-latest"
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


def test_load_migrates_pre_0_1_1_fitted_state_ids(tmp_path: Path) -> None:
    serialized_generator = {
        "config": {
            "random_seed": 42,
            "use_nlp_extractor": False,
            "use_graph_community_detection": False,
            "extraction": {
                "lowercase": True,
                "min_concept_length": 2,
                "min_document_frequency": 1,
                "max_concept_df_ratio": 1.0,
                "max_phrase_length": 4,
                "reject_stopword_concepts": True,
                "reject_url_like_concepts": True,
                "reject_generic_shell_concepts": True,
                "clean_technical_documents": True,
                "strip_urls": True,
                "suppress_section_headers": True,
                "spacy_model_name": "en_core_web_sm",
                "allowed_kinds": ["entity", "noun_phrase"],
            },
            "graph": {"min_edge_weight": 1},
            "community_detection": {"resolution": 1.0, "random_seed": 42},
            "label_assignment": {
                "max_labels_per_paragraph": 3,
                "min_evidence_concepts": 1,
                "min_label_support": 1.0,
            },
            "verbalization": {"top_k_label_terms": 5, "max_display_terms": 3},
        },
        "is_fitted": True,
        "fitted_concepts": [
            {
                "id": "noun_phrase:openai",
                "surface": "OpenAI",
                "normalized": "openai",
                "kind": "noun_phrase",
                "document_frequency": 2,
            },
            {
                "id": "entity:language models",
                "surface": "language models",
                "normalized": "language models",
                "kind": "entity",
                "document_frequency": 2,
            },
        ],
        "fitted_communities": [
            {
                "id": "community-0",
                "concept_ids": ["noun_phrase:openai", "entity:language models"],
                "display_name": "openai / language models",
                "representative_concepts": ["openai", "language models"],
                "size": 2,
            }
        ],
    }
    output_path = tmp_path / "legacy-generator.json"
    output_path.write_text(json.dumps(serialized_generator), encoding="utf-8")

    loaded = LabelGenerator.load(output_path)
    result = loaded.transform(["OpenAI uses language models."])

    assert result.mentions
    assert result.concepts
    assert result.paragraph_labels[0].label_ids == ["community-0"]


def test_config_from_dict_coerces_empty_llm_config() -> None:
    from labelgen.io.serialize import config_from_dict

    loaded = config_from_dict(
        {
            "extractor_mode": "llm",
            "use_graph_community_detection": False,
            "extraction": {
                "llm": {},
            },
        }
    )

    assert loaded.extraction.llm.model == ""
    assert loaded.extraction.llm.provider == "openai"


def test_config_from_dict_coerces_null_llm_config() -> None:
    from labelgen.io.serialize import config_from_dict

    loaded = config_from_dict(
        {
            "extractor_mode": "llm",
            "use_graph_community_detection": False,
            "extraction": {
                "llm": None,
            },
        }
    )

    assert loaded.extraction.llm.model == ""
    assert loaded.extraction.llm.provider == "openai"
