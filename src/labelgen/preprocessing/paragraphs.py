"""Paragraph normalization utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from labelgen.types import Paragraph


def make_paragraph_id(text: str) -> str:
    """Create a deterministic paragraph identifier from text content."""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_paragraphs(paragraphs: Sequence[str | Paragraph]) -> list[Paragraph]:
    """Normalize supported paragraph inputs into Paragraph models."""

    normalized: list[Paragraph] = []
    for item in paragraphs:
        if isinstance(item, Paragraph):
            normalized.append(item)
        else:
            normalized.append(Paragraph(id=make_paragraph_id(item), text=item))
    return normalized
