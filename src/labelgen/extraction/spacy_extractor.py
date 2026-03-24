"""spaCy-backed concept extraction."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterable
from typing import cast

from labelgen.config import ExtractionConfig
from labelgen.extraction.concept_extractor import ConceptExtractor
from labelgen.extraction.normalization import normalize_surface
from labelgen.types import ConceptMention, Paragraph


class SpacyConceptExtractor(ConceptExtractor):
    """Concept extractor backed by spaCy entities and noun chunks.

    This is the default extractor for `0.1.0`. The configured spaCy model must be
    installed before extraction is attempted. `en_core_web_sm` is the recommended
    default, but any compatible installed spaCy pipeline can be selected through
    `ExtractionConfig.spacy_model_name`. Callers that do not want this runtime
    requirement should explicitly disable NLP extraction in `LabelGeneratorConfig`.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._nlp: object | None = None

    def extract(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract entity and noun-phrase mentions from paragraphs."""

        return self._extract_with_spacy(paragraphs)

    def _load_spacy_pipeline(self) -> object:
        """Load the configured spaCy pipeline."""

        try:
            spacy = importlib.import_module("spacy")
        except ImportError as error:
            raise RuntimeError(
                "spaCy is required for the default NLP extractor. "
                "Disable NLP extraction in LabelGeneratorConfig to use the heuristic extractor."
            ) from error

        load = getattr(spacy, "load", None)
        if not callable(load):
            raise RuntimeError("spaCy.load is unavailable in the installed spaCy package.")

        try:
            return load(self._config.spacy_model_name)
        except OSError as error:
            raise RuntimeError(
                f"spaCy model '{self._config.spacy_model_name}' is required for the default "
                "NLP extractor. Install it with "
                f"`python -m spacy download {self._config.spacy_model_name}` or set "
                "`use_nlp_extractor=False` to use the heuristic extractor."
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"spaCy model '{self._config.spacy_model_name}' could not be loaded. "
                "Verify that the installed spaCy package and model are compatible, or set "
                "`use_nlp_extractor=False` to use the heuristic extractor."
            ) from error

    def _extract_with_spacy(self, paragraphs: list[Paragraph]) -> list[ConceptMention]:
        """Extract mentions with spaCy entities and noun chunks."""

        mentions: list[ConceptMention] = []
        if self._nlp is None:
            self._nlp = self._load_spacy_pipeline()
        pipe = getattr(self._nlp, "pipe", None)
        if not callable(pipe):
            raise RuntimeError("The configured spaCy pipeline does not expose a callable pipe().")

        docs = self._iter_objects(pipe(paragraph.text for paragraph in paragraphs))
        for paragraph, doc in zip(paragraphs, docs, strict=True):
            mentions.extend(self._extract_doc_mentions(paragraph.id, doc))
        return mentions

    def _extract_doc_mentions(self, paragraph_id: str, doc: object) -> list[ConceptMention]:
        """Extract spaCy spans as mentions while preserving stable ordering."""

        mentions: list[ConceptMention] = []
        seen: set[tuple[int, int, str]] = set()

        for span in self._iter_objects(getattr(doc, "ents", ())):
            mention = self._make_mention(
                paragraph_id,
                str(getattr(span, "text", "")),
                "entity",
                self._as_int(getattr(span, "start_char", None)),
                self._as_int(getattr(span, "end_char", None)),
            )
            key = (mention.start or 0, mention.end or 0, mention.kind)
            if key not in seen:
                seen.add(key)
                mentions.append(mention)

        noun_chunks = self._iter_noun_chunks(doc)
        for chunk in noun_chunks:
            mention = self._make_mention(
                paragraph_id,
                str(getattr(chunk, "text", "")),
                "noun_phrase",
                self._as_int(getattr(chunk, "start_char", None)),
                self._as_int(getattr(chunk, "end_char", None)),
            )
            key = (mention.start or 0, mention.end or 0, mention.kind)
            if key not in seen:
                seen.add(key)
                mentions.append(mention)

        return sorted(mentions, key=lambda item: (item.start or -1, item.end or -1, item.kind))

    def _iter_noun_chunks(self, doc: object) -> tuple[object, ...]:
        """Return noun chunks if the loaded pipeline supports them."""

        noun_chunks_attribute = "noun_chunks"
        try:
            noun_chunks = getattr(doc, noun_chunks_attribute)
        except (AttributeError, ValueError):
            return ()
        return self._iter_objects(noun_chunks)

    def _make_mention(
        self,
        paragraph_id: str,
        surface: str,
        kind: str,
        start: int,
        end: int,
    ) -> ConceptMention:
        """Create a normalized concept mention."""

        normalized = normalize_surface(surface, lowercase=self._config.lowercase)
        concept_id = self._make_concept_id(normalized, kind)
        return ConceptMention(
            paragraph_id=paragraph_id,
            concept_id=concept_id,
            surface=surface,
            normalized=normalized,
            kind=kind,
            start=start,
            end=end,
        )

    def _make_concept_id(self, normalized: str, kind: str) -> str:
        """Create a stable identifier for a normalized concept."""

        digest = hashlib.sha256(f"{kind}:{normalized}".encode()).hexdigest()
        return digest[:16]

    def _as_int(self, value: object) -> int:
        """Convert known integer-like values to ``int``."""

        if isinstance(value, int):
            return value
        raise TypeError("spaCy span offsets must be integers")

    def _iter_objects(self, value: object) -> tuple[object, ...]:
        """Convert an arbitrary iterable object into a typed tuple boundary."""

        if isinstance(value, Iterable):
            return tuple(cast(Iterable[object], value))
        return ()
