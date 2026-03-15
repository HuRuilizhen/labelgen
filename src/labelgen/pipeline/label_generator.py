"""Main label generation pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from labelgen.community.leiden_detector import LeidenCommunityDetector
from labelgen.config import LabelGeneratorConfig
from labelgen.extraction.filtering import filter_mentions
from labelgen.extraction.normalization import normalize_surface
from labelgen.extraction.spacy_extractor import SpacyConceptExtractor
from labelgen.graph.builder import build_concept_graph
from labelgen.graph.concept_graph import ConceptGraph
from labelgen.io.serialize import dump_config, load_config
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

        artifacts = self._prepare_artifacts(paragraphs)
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

        artifacts = self._prepare_artifacts(paragraphs)
        known_concept_ids = {concept.id for concept in self._fitted_concepts}
        filtered_mentions = [
            mention for mention in artifacts.mentions if mention.concept_id in known_concept_ids
        ]
        concepts = [concept for concept in artifacts.concepts if concept.id in known_concept_ids]
        graph = build_concept_graph(filtered_mentions, self.config.graph)
        return self._build_result(
            paragraphs=artifacts.paragraphs,
            concepts=concepts,
            mentions=filtered_mentions,
            communities=self._fitted_communities,
            graph=graph,
        )

    def fit_transform(self, paragraphs: list[str] | list[Paragraph]) -> LabelGenerationResult:
        """Learn communities and label the same input paragraphs."""

        artifacts = self._prepare_artifacts(paragraphs)
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
        """Persist generator configuration to disk."""

        dump_config(self.config, path)

    @classmethod
    def load(cls, path: str | Path) -> LabelGenerator:
        """Load a generator from serialized configuration."""

        return cls(load_config(path))

    def _prepare_artifacts(
        self,
        paragraphs: list[str] | list[Paragraph],
    ) -> _PipelineArtifacts:
        """Prepare normalized pipeline artifacts from input paragraphs."""

        normalized_paragraphs = normalize_paragraphs(paragraphs)
        extracted_mentions = self._extractor.extract(normalized_paragraphs)
        filtered_mentions = filter_mentions(extracted_mentions, self.config.extraction)
        concepts = self._build_concepts(filtered_mentions)
        retained_concept_ids = self._select_retained_concept_ids(
            concepts,
            len(normalized_paragraphs),
        )
        filtered_mentions = [
            mention for mention in filtered_mentions if mention.concept_id in retained_concept_ids
        ]
        concepts = [concept for concept in concepts if concept.id in retained_concept_ids]
        graph = build_concept_graph(filtered_mentions, self.config.graph)
        return _PipelineArtifacts(
            paragraphs=normalized_paragraphs,
            mentions=filtered_mentions,
            concepts=concepts,
            graph=graph,
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
