"""Tests for paragraph normalization."""

from labelgen.preprocessing.paragraphs import make_paragraph_id, normalize_paragraphs


def test_make_paragraph_id_is_deterministic() -> None:
    assert make_paragraph_id("hello") == make_paragraph_id("hello")


def test_normalize_paragraphs_wraps_strings() -> None:
    paragraphs = normalize_paragraphs(["alpha", "beta"])
    assert len(paragraphs) == 2
    assert paragraphs[0].text == "alpha"
    assert paragraphs[0].id
