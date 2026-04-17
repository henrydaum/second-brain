"""
Token parsing utilities.

Reasoning models (MiniMax M2.7, DeepSeek-R1, QwQ, etc.) emit their
chain-of-thought inside ``<think>…</think>`` or ``<thinking>…</thinking>``
tags. Agent frameworks may also leak XML tool invocations. 

This module provides functions to extract reasoning blocks and
scrub all structural tokens to return clean text for the UI.
"""

import re

# Matches <think>...</think> and <thinking>...</thinking>.
# Opening tag is optional — Qwen models may omit it and only emit </think>.
_THINKING_PATTERN = re.compile(
    r"(?:<(?:think|thinking)>)?(.*?)</(?:think|thinking)>",
    re.DOTALL,
)

# Matches <invoke> blocks, <tool_call> blocks, Minimax tags, and common EOS tokens.
_STRUCTURAL_PATTERN = re.compile(
    r"<invoke.*?>.*?</invoke>|<tool_call.*?>.*?</tool_call>|<(?:/)?minimax:tool_call>|<\|im_end\|>|<\|eot_id\|>",
    re.DOTALL,
)

# Handles malformed or partial thinking tags that arrive without a matching pair,
# e.g. a title response that is only "<think>".
_THINKING_TAG_PATTERN = re.compile(r"</?(?:think|thinking)>")


def strip_model_tokens(text: str) -> tuple[str, list[str]]:
    """Remove thinking blocks and tool call tokens from *text*.

    Returns:
        A ``(clean_text, thinking_blocks)`` tuple where
        *clean_text* has all XML/structural regions removed
        (leading/trailing whitespace stripped), and *thinking_blocks* is a
        list of the extracted inner thoughts (in order of appearance).
    """
    # Extract the thinking content
    blocks = [m.group(1).strip() for m in _THINKING_PATTERN.finditer(text)]
    
    # Strip thinking tags and their content
    clean = _THINKING_PATTERN.sub("", text)
    
    # Strip tool calls and leaked EOS tokens
    clean = _STRUCTURAL_PATTERN.sub("", clean)

    # Strip any leftover unmatched thinking tags.
    clean = _THINKING_TAG_PATTERN.sub("", clean).strip()
    
    return clean, blocks
