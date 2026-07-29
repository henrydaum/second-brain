"""Tests for the streaming token filter (runtime/token_stripper.py).

StreamingTokenFilter is the fragment-by-fragment twin of
``strip_model_tokens``: thinking blocks and EOS tokens must never reach a
frontend's display, even when a tag is split across delta boundaries.
"""

from runtime.token_stripper import StreamingTokenFilter, strip_model_tokens


def _run(fragments):
    f = StreamingTokenFilter()
    out = "".join(f.feed(frag) for frag in fragments)
    return out + f.flush()


def test_plain_text_passes_through():
    assert _run(["Hello ", "there!"]) == "Hello there!"


def test_think_block_is_suppressed():
    assert _run(["<think>secret plan</think>", "Hey! What's up?"]) == "Hey! What's up?"


def test_think_block_split_across_fragments():
    fragments = ["<th", "ink>the user ", "said hey</t", "hink>\n\nHey", "! How can I help?"]
    assert _run(fragments) == "Hey! How can I help?"


def test_thinking_variant_and_interleaved_text():
    assert _run(["Sure — ", "<thinking>hmm</thinking>", "done."]) == "Sure — done."


def test_eos_tokens_are_dropped():
    assert _run(["All done.", "<|im_end|>"]) == "All done."
    assert _run(["All ", "done.<|eot", "_id|>"]) == "All done."


def test_streamed_tool_call_block_is_suppressed():
    raw = (
        'Before<tool_call>\n'
        '<invoke name="grep"><pattern>llm</pattern>'
        '</invoke>\n</tool_call>After'
    )
    fragments = [raw[i:i + 4] for i in range(0, len(raw), 4)]
    assert _run(fragments) == "BeforeAfter"


def test_stray_closer_tag_is_removed():
    # Qwen-style omitted opener: the preceding text streams (accepted
    # limitation) but the closer tag itself never displays.
    assert _run(["reasoning first", "</think>", " answer"]) == "reasoning first answer"


def test_legitimate_angle_bracket_text_survives():
    assert _run(["use List<int> here"]) == "use List<int> here"
    # A '<' tail that never becomes a tag is released by flush.
    assert _run(["a < b and a <t"]) == "a < b and a <t"


def test_leading_whitespace_after_think_is_trimmed():
    out = _run(["<think>x</think>", "\n\n", "Hello"])
    assert out == "Hello"


def test_matches_batch_stripper_on_typical_response():
    raw = "<think>\nThe user said hey.\n</think>\n\nHey! What can I help you with today?"
    clean, _ = strip_model_tokens(raw)
    # Feed in awkward 3-char fragments to stress boundary handling.
    fragments = [raw[i:i + 3] for i in range(0, len(raw), 3)]
    assert _run(fragments) == clean


# ──────────────────────────────────────────────────────────────────────
# The optional-opener regression.
#
# ``_THINKING_PATTERN`` used to make the opening tag optional so a Qwen-style
# omitted opener would still be stripped. Under DOTALL with a non-greedy
# ``.*?`` that means "anything at all, up to the next closer", and ``sub``
# applies it repeatedly — so the text *between* two think blocks matched as an
# unopened block and was deleted. Live damage ranged from a lost opening
# clause to a whole reply arriving as ".".
#
# Each case below is a shape that was seen in the wild or is one edit away
# from one.
# ──────────────────────────────────────────────────────────────────────

def test_an_answer_between_two_thinking_blocks_survives():
    """The reported bug: the front of the reply silently disappeared."""
    raw = ("<think>planning</think>Sure, I can write that for you."
           "<think>more planning</think> Let me first check the docs.")

    clean, blocks = strip_model_tokens(raw)

    assert clean == "Sure, I can write that for you. Let me first check the docs."
    assert blocks == ["planning", "more planning"]


def test_a_thinking_block_after_the_answer_does_not_eat_it():
    """Reply first, reasoning second — this left only the trailing clause."""
    raw = "Sure, I can write that for you<think>x</think>. Let me first check."

    clean, _ = strip_model_tokens(raw)

    assert clean == "Sure, I can write that for you. Let me first check."


def test_a_reply_never_collapses_to_punctuation():
    """The "." messages. Where there *is* an answer, deleting it is never the
    right reading — however the tags fall around it."""
    for raw in ("Hey - still here.\n\n</think>.",
                "Hey - still here. What are we doing?</think>",
                "<think>A</think>Hey - still here.<think>B</think>."):
        clean, _ = strip_model_tokens(raw)
        assert any(c.isalnum() for c in clean), (
            f"{raw!r} was stripped down to {clean!r}")


def test_well_formed_blocks_around_nothing_leave_nothing():
    """The counterpart: when the model really did only emit punctuation
    between its blocks, that is the honest answer and not a symptom."""
    clean, blocks = strip_model_tokens("<think>A</think>. <think>B</think>")

    assert clean == "."
    assert blocks == ["A", "B"]


def test_an_answer_with_a_stray_closer_keeps_the_answer():
    """Only the orphaned tag goes."""
    raw = "Hey - still here. What are we doing?</think>"

    clean, _ = strip_model_tokens(raw)

    assert clean == "Hey - still here. What are we doing?"
    assert "</think>" not in clean


def test_the_qwen_omitted_opener_still_works():
    """The case the optional opener existed for, kept as a narrow rule."""
    raw = "let me work through this</think>Here is the answer."

    clean, blocks = strip_model_tokens(raw)

    assert clean == "Here is the answer."
    assert blocks == ["let me work through this"]


def test_an_omitted_opener_is_not_guessed_beside_a_real_block():
    """A model that tags its reasoning properly is not also omitting openers,
    so a second closer is a stray rather than a second block."""
    raw = "<think>real reasoning</think>The answer.</think>"

    clean, blocks = strip_model_tokens(raw)

    assert clean == "The answer."
    assert blocks == ["real reasoning"]


def test_only_the_first_unopened_closer_is_ever_consumed():
    """Otherwise the rule chews through a reply one closer at a time."""
    raw = "reasoning</think>First part.</think>Second part."

    clean, _ = strip_model_tokens(raw)

    assert "First part." in clean
    assert "Second part." in clean


def test_batch_and_streaming_agree_on_well_formed_blocks():
    """A frontend that shows deltas and then the final text must not watch
    the right answer be replaced by a different one."""
    raw = ("<think>planning</think>Sure, I can help."
           "<think>more</think> Here is how.")

    batch, _ = strip_model_tokens(raw)
    filt = StreamingTokenFilter()
    streamed = "".join(filt.feed(ch) for ch in raw) + filt.flush()

    assert batch.strip() == streamed.strip()
