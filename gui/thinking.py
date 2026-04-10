"""
Thinking token utilities.

Reasoning models (MiniMax M2.7, DeepSeek-R1, QwQ, etc.) emit their
chain-of-thought inside ``<think>…</think>`` or ``<thinking>…</thinking>``
tags.  This module provides a single function to extract those blocks and
return the cleaned text separately.
"""

import re

# Matches <think>…</think> and <thinking>…</thinking>, including
# multiline content.  Uses a non-greedy match so adjacent blocks
# are captured individually rather than merged into one.
_THINKING_PATTERN = re.compile(
    r"<(?:think|thinking)>(.*?)</(?:think|thinking)>",
    re.DOTALL,
)


def strip_thinking(text: str) -> tuple[str, list[str]]:
    """Remove thinking blocks from *text*.

    Returns:
        A ``(clean_text, thinking_blocks)`` tuple where
        *clean_text* has all ``<think>`` / ``<thinking>`` regions removed
        (leading/trailing whitespace stripped), and *thinking_blocks* is a
        list of the extracted inner texts (in order of appearance).

        If no thinking blocks are found, *thinking_blocks* is an empty
        list and *clean_text* is the original text (stripped).
    """
    blocks = [m.group(1).strip() for m in _THINKING_PATTERN.finditer(text)]
    clean = _THINKING_PATTERN.sub("", text).strip()
    return clean, blocks
