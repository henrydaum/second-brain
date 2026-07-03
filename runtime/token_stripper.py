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


class StreamingTokenFilter:
    """Incrementally strip thinking blocks and EOS tokens from streamed text.

    The batch ``strip_model_tokens`` sees the whole response at once; this is
    its streaming twin, fed fragment by fragment. It suppresses everything
    between ``<think>``/``<thinking>`` and the matching closer, drops stray
    closers and leaked EOS tokens, and withholds a fragment's tail when it
    could be the start of a tag split across fragment boundaries (``"<thi"``
    + ``"nk>"``). Leading whitespace is trimmed until the first visible
    output so a response that opens with a thinking block doesn't start the
    display with blank lines.

    Known limitation, accepted for latency: the batch stripper treats text
    before an *unopened* ``</think>`` (Qwen-style omitted opener) as
    thinking; a streaming filter can't know that without buffering the whole
    response, so that text is displayed and only the stray closer tag itself
    is removed.
    """

    _OPENERS = ("<think>", "<thinking>")
    _CLOSERS = ("</think>", "</thinking>")
    _DROPPED = ("<|im_end|>", "<|eot_id|>")
    _ALL_TAGS = _OPENERS + _CLOSERS + _DROPPED

    def __init__(self):
        self._tail = ""
        self._in_think = False
        self._emitted = False

    @classmethod
    def _find_first(cls, text: str, tags: tuple[str, ...]) -> tuple[int | None, str | None]:
        best_idx, best_tag = None, None
        for tag in tags:
            idx = text.find(tag)
            if idx != -1 and (best_idx is None or idx < best_idx):
                best_idx, best_tag = idx, tag
        return best_idx, best_tag

    @classmethod
    def _partial_tag_tail(cls, text: str) -> str:
        """Longest suffix of ``text`` that is a proper prefix of some tag."""
        max_len = min(len(text), max(len(t) for t in cls._ALL_TAGS) - 1)
        for length in range(max_len, 0, -1):
            suffix = text[-length:]
            if any(tag.startswith(suffix) for tag in cls._ALL_TAGS):
                return suffix
        return ""

    def feed(self, fragment: str) -> str:
        """Return the displayable portion of ``fragment`` (possibly empty)."""
        text = self._tail + (fragment or "")
        self._tail = ""
        out: list[str] = []
        while text:
            if self._in_think:
                idx, closer = self._find_first(text, self._CLOSERS)
                if idx is None:
                    self._tail = self._partial_tag_tail(text)
                    text = ""
                else:
                    text = text[idx + len(closer):]
                    self._in_think = False
            else:
                idx, tag = self._find_first(text, self._OPENERS + self._CLOSERS + self._DROPPED)
                if idx is None:
                    keep = self._partial_tag_tail(text)
                    out.append(text[:len(text) - len(keep)] if keep else text)
                    self._tail = keep
                    text = ""
                else:
                    out.append(text[:idx])
                    text = text[idx + len(tag):]
                    self._in_think = tag in self._OPENERS
        emitted = "".join(out)
        if not self._emitted:
            emitted = emitted.lstrip()
        if emitted:
            self._emitted = True
        return emitted

    def flush(self) -> str:
        """Release any withheld tail at end of stream (it wasn't a tag)."""
        tail, self._tail = self._tail, ""
        if self._in_think or not tail:
            return ""
        tail = tail if self._emitted else tail.lstrip()
        if tail:
            self._emitted = True
        return tail


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
