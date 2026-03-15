"""Paragraph normalization utilities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from labelgen.types import Paragraph

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTI_DASH_RE = re.compile(r"-+")


def normalize_paragraph_text(text: str) -> str:
    """Normalize paragraph text for preprocessing and ID generation."""

    return _WHITESPACE_RE.sub(" ", text.strip())


def normalize_title_for_id(title: str) -> str:
    """Normalize a title into a readable, stable identifier component."""

    lowered = normalize_paragraph_text(title).lower()
    dashed = _NON_ALNUM_RE.sub("-", lowered)
    compact = _MULTI_DASH_RE.sub("-", dashed).strip("-")
    return compact or "paragraph"


def build_paragraph_id(paragraph: Paragraph, index: int) -> str:
    """Build a deterministic paragraph identifier with precedence-aware rules."""

    if paragraph.id:
        return paragraph.id

    metadata = paragraph.metadata or {}
    if doc_id := _get_string_metadata(metadata, "doc_id"):
        return f"{doc_id}#p{index}"
    if title := _get_string_metadata(metadata, "title"):
        return f"{normalize_title_for_id(title)}#p{index}"

    normalized_text = normalize_paragraph_text(paragraph.text)
    digest = hashlib.sha256(normalized_text.encode()).hexdigest()[:12]
    return f"p{index}-{digest}"


def normalize_paragraphs(paragraphs: Sequence[str | Paragraph]) -> list[Paragraph]:
    """Normalize supported paragraph inputs into Paragraph models."""

    normalized: list[Paragraph] = []
    for raw_index, item in enumerate(paragraphs):
        paragraph = _coerce_paragraph(item)
        text = normalize_paragraph_text(paragraph.text)
        if not text:
            continue

        normalized.append(
            Paragraph(
                id=build_paragraph_id(
                    Paragraph(
                        id=paragraph.id,
                        text=text,
                        metadata=paragraph.metadata,
                    ),
                    raw_index,
                ),
                text=text,
                metadata=paragraph.metadata,
            )
        )
    return normalized


def _coerce_paragraph(item: str | Paragraph) -> Paragraph:
    """Wrap supported input types into a paragraph model."""

    if isinstance(item, Paragraph):
        return item
    return Paragraph(id="", text=item)


def _get_string_metadata(metadata: dict[str, Any], key: str) -> str | None:
    """Return a stripped string metadata value when present."""

    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    normalized = normalize_paragraph_text(value)
    return normalized or None
