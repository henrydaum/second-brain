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
