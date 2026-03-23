"""Technical-document cleanup helpers."""

from __future__ import annotations

import re

from labelgen.config import ExtractionConfig
from labelgen.types import Paragraph

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_BANNER_RE = re.compile(r"(?:[\*\-=#_]){4,}")
_SECTION_HEADER_RE = re.compile(
    r"\b(?:"
    r"problem summary|problem description|problem conclusion|fix information|"
    r"subscribe to this apar|apar status|error description|users affected|"
    r"direct links to fixes|references"
    r")\b:?",
    re.IGNORECASE,
)
_EXTRA_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:])")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_paragraphs(
    paragraphs: list[Paragraph],
    config: ExtractionConfig,
) -> list[Paragraph]:
    """Apply support-document cleanup rules before concept extraction."""

    if not config.clean_technical_documents:
        return paragraphs

    cleaned: list[Paragraph] = []
    for paragraph in paragraphs:
        text = clean_paragraph_text(paragraph.text, config)
        cleaned.append(Paragraph(id=paragraph.id, text=text, metadata=paragraph.metadata))
    return cleaned


def clean_paragraph_text(text: str, config: ExtractionConfig) -> str:
    """Clean technical-document artifacts from paragraph text."""

    cleaned = _BANNER_RE.sub(" ", text)
    if config.strip_urls:
        cleaned = _URL_RE.sub(" ", cleaned)
    if config.suppress_section_headers:
        cleaned = _SECTION_HEADER_RE.sub(" ", cleaned)
    cleaned = _EXTRA_PUNCT_SPACE_RE.sub(r"\1", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip(" -:;,.")
    return cleaned
