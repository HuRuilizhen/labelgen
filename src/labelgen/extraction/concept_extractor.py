"""Abstract concept extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from labelgen.types import ConceptMention, Paragraph


class ConceptExtractor(ABC):
    """Abstract base class for paragraph concept extraction."""

    @abstractmethod
    def extract(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract concept mentions from paragraphs."""
