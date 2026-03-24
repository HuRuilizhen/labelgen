"""Tests for technical-document cleanup."""

from labelgen.config import ExtractionConfig
from labelgen.preprocessing.cleanup import clean_paragraph_text, clean_paragraphs
from labelgen.types import Paragraph


def test_clean_paragraph_text_removes_urls_banners_and_section_headers() -> None:
    config = ExtractionConfig()
    text = (
        "PROBLEM DESCRIPTION: ******** Visit https://example.com/path for details. "
        "ERROR DESCRIPTION: Actual service failure follows."
    )

    cleaned = clean_paragraph_text(text, config)

    assert "https://example.com/path" not in cleaned
    assert "PROBLEM DESCRIPTION" not in cleaned
    assert "ERROR DESCRIPTION" not in cleaned
    assert "Actual service failure follows" in cleaned


def test_clean_paragraphs_preserves_ids_and_metadata() -> None:
    paragraph = Paragraph(
        id="doc-1#p0",
        text="FIX INFORMATION: See www.ibm.com/support and proceed.",
        metadata={"doc_id": "doc-1"},
    )

    cleaned = clean_paragraphs([paragraph], ExtractionConfig())

    assert cleaned[0].id == "doc-1#p0"
    assert cleaned[0].metadata == {"doc_id": "doc-1"}
    assert "www.ibm.com/support" not in cleaned[0].text


def test_clean_paragraphs_can_be_disabled() -> None:
    config = ExtractionConfig(clean_technical_documents=False)
    paragraph = Paragraph(id="p1", text="PROBLEM SUMMARY: Keep https://example.com")

    cleaned = clean_paragraphs([paragraph], config)

    assert cleaned[0].text == "PROBLEM SUMMARY: Keep https://example.com"


def test_clean_paragraph_text_removes_subscribe_boilerplate_and_markup_tokens() -> None:
    config = ExtractionConfig()
    text = "SUBSCRIBE You can track all active APARs for this component [ [ * *"

    cleaned = clean_paragraph_text(text, config)

    assert cleaned == ""
