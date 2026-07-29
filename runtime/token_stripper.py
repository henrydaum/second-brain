"""
Token parsing utilities.

Reasoning models (MiniMax M2.7, DeepSeek-R1, QwQ, etc.) emit their
chain-of-thought inside ``<think>…</think>`` or ``<thinking>…</thinking>``
tags. Agent frameworks may also leak XML tool invocations. 

This module provides functions to extract reasoning blocks and
scrub all structural tokens to return clean text for the UI.
"""

import re

# A properly opened reasoning block. The opening tag is *required*, and that
# is the whole of a bug that silently deleted people's replies.
#
# It used to be optional, so that a Qwen-style omitted opener would still be
# stripped. But an optional opener in front of a non-greedy `.*?` under DOTALL
# means the pattern matches "anything at all, up to the next closer" — and
# ``sub`` applies it repeatedly. Two think blocks around a sentence therefore
# ate the sentence: after the first block is consumed, the scan resumes at the
# answer, matches it as an unopened block running to the *second* closer, and
# removes it. One block placed after the answer did the same thing to
# everything before it. The observed damage ranged from a lost opening clause
# to a reply that arrived as a single ".".
#
# The Qwen case is still handled, but as the narrow thing it is: see
# ``_strip_leading_unopened``.
_THINKING_PATTERN = re.compile(
    r"<(?:think|thinking)>(.*?)</(?:think|thinking)>",
    re.DOTALL,
)

# One closing tag, for the unopened case.
_CLOSER = re.compile(r"</(?:think|thinking)>")

# Matches <invoke> blocks, <tool_call> blocks, Minimax tags, and common EOS tokens.
_STRUCTURAL_PATTERN = re.compile(
    r"<invoke.*?>.*?</invoke>|<tool_call.*?>.*?</tool_call>|<(?:/)?minimax:tool_call>|"
    r"<\|im_end\|>|<\|eot_id\|>",
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
    _TOOL_OPENERS = ("<tool_call>",)
    _TOOL_CLOSERS = ("</tool_call>",)
    _DROPPED = ("<|im_end|>", "<|eot_id|>")
    _ALL_TAGS = (
        _OPENERS + _CLOSERS + _TOOL_OPENERS + _TOOL_CLOSERS + _DROPPED
    )

    def __init__(self):
        self._tail = ""
        self._in_think = False
        self._in_tool = False
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
            if self._in_think or self._in_tool:
                closers = self._CLOSERS if self._in_think else self._TOOL_CLOSERS
                idx, closer = self._find_first(text, closers)
                if idx is None:
                    self._tail = self._partial_tag_tail(text)
                    text = ""
                else:
                    text = text[idx + len(closer):]
                    self._in_think = False
                    self._in_tool = False
            else:
                tags = (
                    self._OPENERS + self._CLOSERS
                    + self._TOOL_OPENERS + self._TOOL_CLOSERS + self._DROPPED
                )
                idx, tag = self._find_first(text, tags)
                if idx is None:
                    keep = self._partial_tag_tail(text)
                    out.append(text[:len(text) - len(keep)] if keep else text)
                    self._tail = keep
                    text = ""
                else:
                    out.append(text[:idx])
                    text = text[idx + len(tag):]
                    self._in_think = tag in self._OPENERS
                    self._in_tool = tag in self._TOOL_OPENERS
        emitted = "".join(out)
        if not self._emitted:
            emitted = emitted.lstrip()
        if emitted:
            self._emitted = True
        return emitted

    def flush(self) -> str:
        """Release any withheld tail at end of stream (it wasn't a tag)."""
        tail, self._tail = self._tail, ""
        if self._in_think or self._in_tool or not tail:
            return ""
        tail = tail if self._emitted else tail.lstrip()
        if tail:
            self._emitted = True
        return tail


def _is_substantive(text: str) -> bool:
    """Whether text reads as an actual reply rather than leftover punctuation.

    The question this answers is "if I treat the head as reasoning, is what
    remains something a person was meant to read?" — and the failures it
    exists to prevent both left behind exactly one character.
    """
    return any(char.isalnum() for char in text)


def _strip_leading_unopened(text: str) -> tuple[str, str | None]:
    """Handle a closing tag whose opener the model never emitted.

    Some models (Qwen-style) stream reasoning and close it without ever
    opening it, so the head of the response is thinking and the tag is the
    only marker. That reading is a guess, and it is unfalsifiable in general
    — a stray ``</think>`` in the middle of a genuine answer looks identical.

    So it is applied under two conditions, both of which the damage taught us.
    Only the **first** closer counts, so this can never chew through a reply
    one block at a time. And it only applies if what follows is a substantive
    reply — because the reading that deletes the entire answer and leaves a
    full stop is never the right one, whatever the tags say.

    Returns ``(text, extracted_head_or_None)``.
    """
    match = _CLOSER.search(text)
    if match is None:
        return text, None
    head, tail = text[:match.start()], text[match.end():]
    if not _is_substantive(tail):
        return text, None
    return tail, head.strip()


def strip_model_tokens(text: str) -> tuple[str, list[str]]:
    """Remove thinking blocks and tool call tokens from *text*.

    Returns:
        A ``(clean_text, thinking_blocks)`` tuple where
        *clean_text* has all XML/structural regions removed
        (leading/trailing whitespace stripped), and *thinking_blocks* is a
        list of the extracted inner thoughts (in order of appearance).

    The ordering matters: properly opened blocks are removed first, so that
    what reaches the unopened-closer guess is only a tag with no partner.
    """
    blocks = [m.group(1).strip() for m in _THINKING_PATTERN.finditer(text)]
    clean = _THINKING_PATTERN.sub("", text)

    # Only when the model emitted no well-formed block at all. One that tags
    # its reasoning properly is not also omitting openers, so a stray closer
    # alongside a real block is a stray, not a second block — and guessing
    # otherwise is how a reply loses its opening sentence.
    if not blocks:
        clean, unopened = _strip_leading_unopened(clean)
        if unopened:
            blocks.append(unopened)

    # Strip tool calls and leaked EOS tokens
    clean = _STRUCTURAL_PATTERN.sub("", clean)

    # Strip any leftover unmatched thinking tags. Reaching here means the
    # guess above declined, so the answer is kept and only the tag goes.
    clean = _THINKING_TAG_PATTERN.sub("", clean).strip()

    return clean, blocks
