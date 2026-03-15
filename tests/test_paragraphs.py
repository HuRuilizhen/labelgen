"""Tests for paragraph normalization."""

from labelgen.preprocessing.paragraphs import (
    build_paragraph_id,
    normalize_paragraph_text,
    normalize_paragraphs,
    normalize_title_for_id,
)
from labelgen.types import Paragraph


def test_normalize_paragraph_text_strips_and_collapses_whitespace() -> None:
    assert normalize_paragraph_text("  hello\n\nworld  ") == "hello world"


def test_build_paragraph_id_is_deterministic_and_position_aware() -> None:
    paragraph = Paragraph(id="", text="hello", metadata=None)

    assert build_paragraph_id(paragraph, 0) == build_paragraph_id(paragraph, 0)
    assert build_paragraph_id(paragraph, 0) != build_paragraph_id(paragraph, 1)


def test_build_paragraph_id_prefers_existing_id_then_doc_id_then_title() -> None:
    explicit = Paragraph(id="custom-id", text="alpha", metadata={"doc_id": "doc-a"})
    by_doc = Paragraph(id="", text="alpha", metadata={"doc_id": "doc-a"})
    by_title = Paragraph(id="", text="alpha", metadata={"title": "Project Notes"})

    assert build_paragraph_id(explicit, 0) == "custom-id"
    assert build_paragraph_id(by_doc, 2) == "doc-a#p2"
    assert build_paragraph_id(by_title, 3) == "project-notes#p3"


def test_normalize_title_for_id_creates_readable_slug() -> None:
    assert normalize_title_for_id("  A Sample: Title!  ") == "a-sample-title"


def test_normalize_paragraphs_wraps_strings_and_drops_empty_entries() -> None:
    paragraphs = normalize_paragraphs(["  alpha  ", "   ", "\n beta\t"])

    assert len(paragraphs) == 2
    assert [paragraph.text for paragraph in paragraphs] == ["alpha", "beta"]
    assert paragraphs[0].id != paragraphs[1].id


def test_normalize_paragraphs_preserves_order_and_distinguishes_repeated_text() -> None:
    paragraphs = normalize_paragraphs(["Repeated text", "Repeated text"])

    assert [paragraph.text for paragraph in paragraphs] == ["Repeated text", "Repeated text"]
    assert paragraphs[0].id != paragraphs[1].id


def test_normalize_paragraphs_preserves_caller_provided_ids() -> None:
    paragraphs = normalize_paragraphs(
        [
            Paragraph(id="keep-me", text="  alpha  "),
            Paragraph(id="", text=" beta ", metadata={"doc_id": "doc-1"}),
        ]
    )

    assert [paragraph.id for paragraph in paragraphs] == ["keep-me", "doc-1#p1"]
    assert [paragraph.text for paragraph in paragraphs] == ["alpha", "beta"]
