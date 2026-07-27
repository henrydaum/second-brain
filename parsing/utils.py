"""Shared text utilities for parser helpers.

Kernel code, alongside :class:`~parsing.result.ParseResult`, and re-exported
from :mod:`parsing` so every parser reaches it the same way::

    from parsing import clean_text, max_chars

One stable absolute path matters here: a parser package physically lands in
whichever tree it was installed into, and a relative import would resolve
against *that* tree rather than the kernel. Keep it dependency-free (stdlib
only) — every parser, heavy or light, relies on it.
"""

import re

# ~125k tokens. Generous default ceiling for text extraction.
DEFAULT_MAX_CHARS = 500_000


def max_chars(config: dict | None) -> int:
    """Return the configured character limit for text parsing."""
    return (config or {}).get("max_chars", DEFAULT_MAX_CHARS)


def clean_text(text: str, preserve_indent: bool = False) -> str:
    """Normalize whitespace and remove junk.

    If preserve_indent is True, only collapse horizontal whitespace within
    lines (not leading whitespace), keeping indentation intact.
    """
    if not text:
        return ""
    if preserve_indent:
        # Collapse runs of spaces/tabs mid-line only, keep leading whitespace
        text = re.sub(r"(?<=\S)[ \t]+", " ", text)
    else:
        text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
