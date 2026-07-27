"""Shared text utilities for parser helpers.

This module lives permanently in the kernel (``plugins/services/helpers``),
so parser *packages* can import it with a stable absolute path
(``from plugins.services.helpers.parsing_utils import clean_text``) no matter
which tree their ``parse_*.py`` physically lands in. Keep it dependency-free
(stdlib only) — every parser, heavy or light, relies on it.
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
