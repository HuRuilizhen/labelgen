"""Main label generation pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from labelgen.community.leiden_detector import LeidenCommunityDetector
from labelgen.config import LabelGeneratorConfig
from labelgen.extraction.filtering import filter_mentions
from labelgen.extraction.normalization import normalize_surface
from labelgen.extraction.spacy_extractor import SpacyConceptExtractor
from labelgen.graph.builder import build_concept_graph
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.io.serialize import dump_json_object, load_json_object
from labelgen.labeling.assigner import assign_paragraph_labels
from labelgen.labeling.verbalizer import verbalize_communities
from labelgen.preprocessing.paragraphs import normalize_paragraphs
from labelgen.types import (
    Community,
    Concept,
    ConceptMention,
    GraphSummary,
    LabelGenerationResult,
    Paragraph,
    ParagraphLabels,
)


@dataclass(slots=True)
class _PipelineArtifacts:
    """Normalized pipeline artifacts used across fit and transform."""

    paragraphs: list[Paragraph]
    mentions: list[ConceptMention]
    concepts: list[Concept]
    graph: ConceptGraph


class LabelGenerator:
    """Library entrypoint for label generation."""

    def __init__(self, config: LabelGeneratorConfig | None = None) -> None:
        self.config = config or LabelGeneratorConfig()
        self._extractor = SpacyConceptExtractor(self.config.extraction)
        self._detector = LeidenCommunityDetector(self.config.community_detection)
        self._is_fitted = False
        self._fitted_communities: list[Community] = []
        self._fitted_concepts: list[Concept] = []

    def fit(self, paragraphs: list[str] | list[Paragraph]) -> LabelGenerator:
        """Learn concept communities from the provided paragraphs."""

        artifacts = self._extract_artifacts(paragraphs)
        retained_concept_ids = self._select_retained_concept_ids(
            artifacts.concepts,
            len(artifacts.paragraphs),
        )
        artifacts = self._retain_concepts(artifacts, retained_concept_ids)
        self._fitted_concepts = artifacts.concepts
        self._fitted_communities = verbalize_communities(
            self._detector.detect(artifacts.graph),
            artifacts.concepts,
            artifacts.graph,
            self.config.verbalization,
        )
        self._is_fitted = True
        return self

    def transform(self, paragraphs: list[str] | list[Paragraph]) -> LabelGenerationResult:
        """Assign labels to paragraphs using previously learned communities."""

        if not self._is_fitted:
            raise RuntimeError("LabelGenerator.transform() requires a fitted generator.")

        artifacts = self._extract_artifacts(paragraphs)
        known_concept_ids = {concept.id for concept in self._fitted_concepts}
        artifacts = self._retain_concepts(artifacts, known_concept_ids)
        return self._build_result(
            paragraphs=artifacts.paragraphs,
            concepts=artifacts.concepts,
            mentions=artifacts.mentions,
            communities=self._fitted_communities,
            graph=artifacts.graph,
        )

    def fit_transform(self, paragraphs: list[str] | list[Paragraph]) -> LabelGenerationResult:
        """Learn communities and label the same input paragraphs."""

        artifacts = self._extract_artifacts(paragraphs)
        retained_concept_ids = self._select_retained_concept_ids(
            artifacts.concepts,
            len(artifacts.paragraphs),
        )
        artifacts = self._retain_concepts(artifacts, retained_concept_ids)
        communities = verbalize_communities(
            self._detector.detect(artifacts.graph),
            artifacts.concepts,
            artifacts.graph,
            self.config.verbalization,
        )
        self._fitted_concepts = artifacts.concepts
        self._fitted_communities = communities
        self._is_fitted = True
        paragraph_labels = assign_paragraph_labels(
            artifacts.paragraphs,
            artifacts.mentions,
            communities,
            self.config.label_assignment,
        )
        return self._build_result(
            paragraphs=artifacts.paragraphs,
            concepts=artifacts.concepts,
            mentions=artifacts.mentions,
            communities=communities,
            graph=artifacts.graph,
            paragraph_labels=paragraph_labels,
        )

    def save(self, path: str | Path) -> None:
        """Persist generator configuration and fitted state to disk."""

        dump_json_object(self._to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> LabelGenerator:
        """Load a generator from serialized configuration and fitted state."""

        data = load_json_object(path)
        return cls._from_dict(data)

    def _extract_artifacts(
        self,
        paragraphs: list[str] | list[Paragraph],
    ) -> _PipelineArtifacts:
        """Extract normalized pipeline artifacts from input paragraphs."""

        normalized_paragraphs = normalize_paragraphs(paragraphs)
        extracted_mentions = self._extractor.extract(normalized_paragraphs)
        filtered_mentions = filter_mentions(extracted_mentions, self.config.extraction)
        concepts = self._build_concepts(filtered_mentions)
        return _PipelineArtifacts(
            paragraphs=normalized_paragraphs,
            mentions=filtered_mentions,
            concepts=concepts,
            graph=build_concept_graph(filtered_mentions, self.config.graph),
        )

    def _retain_concepts(
        self,
        artifacts: _PipelineArtifacts,
        retained_concept_ids: set[str],
    ) -> _PipelineArtifacts:
        """Retain only the requested concepts and rebuild dependent artifacts."""

        mentions = [
            mention for mention in artifacts.mentions if mention.concept_id in retained_concept_ids
        ]
        concepts = [concept for concept in artifacts.concepts if concept.id in retained_concept_ids]
        return _PipelineArtifacts(
            paragraphs=artifacts.paragraphs,
            mentions=mentions,
            concepts=concepts,
            graph=build_concept_graph(mentions, self.config.graph),
        )

    def _build_result(
        self,
        *,
        paragraphs: list[Paragraph],
        concepts: list[Concept],
        mentions: list[ConceptMention],
        communities: list[Community],
        graph: ConceptGraph,
        paragraph_labels: list[ParagraphLabels] | None = None,
    ) -> LabelGenerationResult:
        """Build a structured result object."""

        if paragraph_labels is None:
            paragraph_labels = assign_paragraph_labels(
                paragraphs,
                mentions,
                communities,
                self.config.label_assignment,
            )
        return LabelGenerationResult(
            paragraphs=paragraphs,
            concepts=concepts,
            mentions=mentions,
            communities=communities,
            paragraph_labels=paragraph_labels,
            graph_summary=GraphSummary(
                node_count=graph.node_count,
                edge_count=graph.edge_count,
                metadata={
                    "config": asdict(self.config.graph),
                    "community_detection": asdict(self.config.community_detection),
                    "verbalization": asdict(self.config.verbalization),
                },
            ),
            metadata={
                "random_seed": self.config.random_seed,
                "is_fitted": self._is_fitted,
            },
        )

    def _to_dict(self) -> dict[str, Any]:
        """Serialize generator configuration and fitted state."""

        return {
            "config": asdict(self.config),
            "is_fitted": self._is_fitted,
            "fitted_concepts": [asdict(concept) for concept in self._fitted_concepts],
            "fitted_communities": [asdict(community) for community in self._fitted_communities],
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> LabelGenerator:
        """Reconstruct a generator from serialized data."""

        config_data = data.get("config")
        if not isinstance(config_data, dict):
            raise TypeError("Serialized generator must contain a config object.")

        from labelgen.io.serialize import (
            as_json_object,
            as_json_object_list,
            config_from_dict,
        )

        generator = cls(config_from_dict(as_json_object(cast(object, config_data))))
        is_fitted = data.get("is_fitted", False)
        if not isinstance(is_fitted, bool):
            raise TypeError("Serialized generator is_fitted must be a boolean.")
        generator._is_fitted = is_fitted

        fitted_concepts = as_json_object_list(data.get("fitted_concepts", []))
        generator._fitted_concepts = [
            Concept(**item) for item in fitted_concepts
        ]

        fitted_communities = as_json_object_list(data.get("fitted_communities", []))
        generator._fitted_communities = [
            Community(**item) for item in fitted_communities
        ]

        if generator._is_fitted and (
            not generator._fitted_concepts or not generator._fitted_communities
        ):
            raise TypeError("Serialized fitted generator must include concepts and communities.")
        return generator

    def _select_retained_concept_ids(
        self,
        concepts: list[Concept],
        paragraph_count: int,
    ) -> set[str]:
        """Select concepts that satisfy configured frequency constraints."""

        if paragraph_count == 0:
            return set()

        retained: set[str] = set()
        for concept in concepts:
            document_frequency = concept.document_frequency or 0
            if document_frequency < self.config.extraction.min_document_frequency:
                continue
            if document_frequency / paragraph_count > self.config.extraction.max_concept_df_ratio:
                continue
            retained.add(concept.id)
        return retained

    def _build_concepts(self, mentions: list[ConceptMention]) -> list[Concept]:
        """Aggregate concept mentions into concept models."""

        frequencies: Counter[str] = Counter()
        prototypes: dict[str, ConceptMention] = {}
        paragraph_membership: dict[str, set[str]] = {}

        for mention in mentions:
            prototypes.setdefault(mention.concept_id, mention)
            frequencies[mention.concept_id] += 1
            paragraph_membership.setdefault(mention.concept_id, set()).add(mention.paragraph_id)

        concepts: list[Concept] = []
        for concept_id in sorted(frequencies):
            mention = prototypes[concept_id]
            concepts.append(
                Concept(
                    id=concept_id,
                    surface=mention.surface,
                    normalized=normalize_surface(
                        mention.normalized,
                        lowercase=self.config.extraction.lowercase,
                    ),
                    kind=mention.kind,
                    document_frequency=len(paragraph_membership[concept_id]),
                )
            )
        return concepts
