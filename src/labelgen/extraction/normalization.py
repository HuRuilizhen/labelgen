"""Normalization helpers for extracted concepts."""

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_surface(text: str, *, lowercase: bool = True) -> str:
    """Normalize concept surface text into a stable lookup form."""

    value = _WHITESPACE_RE.sub(" ", text.strip())
    if lowercase:
        value = value.lower()
    return value

