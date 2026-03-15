"""Main label generation pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

from labelgen.community.leiden_detector import LeidenCommunityDetector
from labelgen.config import LabelGeneratorConfig
from labelgen.extraction.filtering import filter_mentions
from labelgen.extraction.normalization import normalize_surface
from labelgen.extraction.spacy_extractor import SpacyConceptExtractor
from labelgen.graph.builder import build_concept_graph
from labelgen.io.serialize import dump_config, load_config
from labelgen.labeling.assigner import assign_paragraph_labels
from labelgen.labeling.verbalizer import verbalize_communities
from labelgen.preprocessing.paragraphs import normalize_paragraphs
from labelgen.types import (
    Concept,
    ConceptMention,
    GraphSummary,
    LabelGenerationResult,
    Paragraph,
)


class LabelGenerator:
    """Library entrypoint for label generation."""

    def __init__(self, config: LabelGeneratorConfig | None = None) -> None:
        self.config = config or LabelGeneratorConfig()
        self._extractor = SpacyConceptExtractor(self.config.extraction)
        self._detector = LeidenCommunityDetector(self.config.community_detection)
        self._is_fitted = False

    def fit(self, paragraphs: list[str] | list[Paragraph]) -> LabelGenerator:
        """Fit pipeline state on the provided paragraphs."""

        self.fit_transform(paragraphs)
        return self

    def transform(self, paragraphs: list[str] | list[Paragraph]) -> LabelGenerationResult:
        """Transform paragraphs into labels using the configured pipeline."""

        return self.fit_transform(paragraphs)

    def fit_transform(self, paragraphs: list[str] | list[Paragraph]) -> LabelGenerationResult:
        """Run the full baseline pipeline and return structured results."""

        normalized_paragraphs = normalize_paragraphs(paragraphs)
        extracted_mentions = self._extractor.extract(normalized_paragraphs)
        filtered_mentions = filter_mentions(extracted_mentions, self.config.extraction)
        concepts = self._build_concepts(filtered_mentions)
        retained_concept_ids = {
            concept.id
            for concept in concepts
            if (concept.document_frequency or 0) >= self.config.extraction.min_document_frequency
        }
        filtered_mentions = [
            mention for mention in filtered_mentions if mention.concept_id in retained_concept_ids
        ]
        concepts = [concept for concept in concepts if concept.id in retained_concept_ids]
        graph = build_concept_graph(filtered_mentions, self.config.graph)
        communities = verbalize_communities(self._detector.detect(graph), concepts)
        paragraph_labels = assign_paragraph_labels(
            normalized_paragraphs,
            filtered_mentions,
            communities,
            self.config.label_assignment,
        )

        self._is_fitted = True
        return LabelGenerationResult(
            paragraphs=normalized_paragraphs,
            concepts=concepts,
            mentions=filtered_mentions,
            communities=communities,
            paragraph_labels=paragraph_labels,
            graph_summary=GraphSummary(
                node_count=graph.node_count,
                edge_count=graph.edge_count,
                metadata={
                    "config": asdict(self.config.graph),
                    "community_detection": asdict(self.config.community_detection),
                },
            ),
            metadata={"random_seed": self.config.random_seed},
        )

    def save(self, path: str | Path) -> None:
        """Persist generator configuration to disk."""

        dump_config(self.config, path)

    @classmethod
    def load(cls, path: str | Path) -> LabelGenerator:
        """Load a generator from serialized configuration."""

        return cls(load_config(path))

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
