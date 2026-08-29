"""What the kernel reads off the store's LLM backend.

Same shape as ``test_store_frontend_contracts`` and
``test_store_attachment_tools``: a kernel invariant that happens to be *about*
a store file. The subject is the kernel's own verdict — what deadline does this
box get — and the store file is the input.

Skips cleanly when no store ref is reachable.
"""

from pathlib import Path

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source

BACKEND = "llm/llm_litellm.py"


def _declarations() -> dict:
    from sandbox.validator import validate

    text = store_source(BACKEND)
    if text is None:
        pytest.skip(f"{BACKEND} is not present on a local store ref")
    return validate(text, filename=Path(BACKEND).name).declarations


def test_the_backend_declares_a_deadline_a_real_generation_fits_inside():
    """The default is 60s, and this is the one plugin it is wrong for.

    A deadline measures *running* time and discounts only what the guest spends
    waiting on **the kernel**. This box waits on a provider's socket inside
    litellm, which counts in full, and streaming does not help — ``llm.delta``
    is a one-way notice, so a box emitting tokens for two minutes accrues two
    minutes of running time.

    Undeclared, a long answer is killed mid-sentence and surfaces as
    ``box 'llm_..._0' died during '__chat__'`` — which names no cause, points
    at no fix, and clears up by itself on the next call because the pool opens
    a fresh box. That is why this is pinned rather than left to whoever reads
    the declaration block: losing it costs a debugging session, not a test.
    """
    from sandbox.interpreter import DEFAULT_TIMEOUT_SECONDS, clamp_timeout

    declared = _declarations().get("timeout")
    assert declared, (
        f"{BACKEND} declares no timeout, so every model call is killed at "
        f"{DEFAULT_TIMEOUT_SECONDS}s of running time")
    assert clamp_timeout(declared) > DEFAULT_TIMEOUT_SECONDS


def test_the_declared_deadline_survives_the_kernel_clamp():
    """Declarations are intent; the kernel clamps them.

    Asking for more than ``MAX_TIMEOUT_SECONDS`` is not an error and not a
    grant — it silently becomes the ceiling. Pinning the *resolved* number is
    the only way to state what a call actually gets.
    """
    from sandbox.interpreter import MAX_TIMEOUT_SECONDS, clamp_timeout

    resolved = clamp_timeout(_declarations().get("timeout"))
    assert resolved == min(600.0, MAX_TIMEOUT_SECONDS)


def test_wall_clock_still_bounds_a_call_that_the_declaration_cannot():
    """The limit that stays, so nobody debugs this twice.

    ``HARD_CEILING`` is wall clock, is not declarable, and is enforced by the
    same watchdog — so one model call over ten minutes dies exactly the way
    the 60s deadline used to kill one over a minute, and raising the
    declaration cannot help. Stated here because the fix above looks like it
    removed the whole class of failure and did not.
    """
    from sandbox.interpreter import clamp_timeout
    from sandbox.watchdog import HARD_CEILING

    assert clamp_timeout(_declarations().get("timeout")) <= HARD_CEILING


# ──────────────────────────────────────────────────────────────────────
# Saying what a parameter is.
# ──────────────────────────────────────────────────────────────────────

def _backend():
    """The backend class, executed out of the store's source.

    Executed rather than validated as text, because what is under test here is
    a *transformation* — docstrings in, prose out — and the way to be sure it
    works is to run it. Nothing here reaches litellm: the provider library is
    imported in ``start``, and these are the pieces that need no provider.
    """
    text = store_source(BACKEND)
    if text is None:
        pytest.skip(f"{BACKEND} is not present on a local store ref")
    namespace: dict = {"__name__": "_store_llm_litellm"}
    exec(compile(text, BACKEND, "exec"), namespace)   # noqa: S102
    return namespace["LiteLLMBackend"]


def test_a_field_docstring_is_read_out_of_the_source(tmp_path):
    """Python discards attribute docstrings — they are a convention tooling
    reads from source, and there is nothing on the class to look them up on.
    So this is an AST pass, and the thing it must get right is *scope*: the
    same module holds nested dicts whose fields are called ``name`` and
    ``description``, and merging those in would attach a sentence about a
    function's name to a parameter called name."""
    import importlib.util
    import textwrap

    path = tmp_path / "_fake_spec.py"
    path.write_text(textwrap.dedent('''
        class Wanted:
            temperature: float
            """What sampling temperature to use."""
            seed: int

        class Nested:
            temperature: str
            """A different thing entirely."""
    '''), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_fake_spec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    found = _backend()._field_docs(module, "Wanted")

    assert found == {"temperature": "What sampling temperature to use."}
    # No docstring under it, so no entry — not an empty string to render.
    assert "seed" not in found
    # A class that is not there is silence, not a crash.
    assert _backend()._field_docs(module, "Absent") == {}


def test_prose_is_trimmed_to_something_a_quote_block_can_hold():
    """Three rules, each earning its place. A hundred-character docs URL is
    most of a terminal line and none of the meaning. The per-model bullets
    OpenAI's fields tail off into are about somebody else's models. And the
    cap lands on a sentence boundary, because a description cut mid-clause
    reads as a bug in the thing showing it."""
    backend = _backend()

    assert backend._tidy(
        "Constrains effort on [reasoning models](https://example.com/x).\n"
        "\n"
        "Reducing it is faster.\n"
        "\n"
        "- `gpt-5.1` defaults to `none`.\n"
    ) == "Constrains effort on reasoning models. Reducing it is faster."

    # ``Learn more`` was a link label; the sentence is empty once the URL has
    # gone, and an instruction with nowhere to go is worse than silence.
    assert backend._tidy(
        "Parameters for audio output. [Learn more](https://example.com)."
    ) == "Parameters for audio output."

    long = ("Sentence one is here. " + "y " * 90
            + "And here is the last whole sentence. " + "z " * 90)
    trimmed = backend._tidy(long)
    assert len(trimmed) <= 300
    assert trimmed.endswith("And here is the last whole sentence.")

    # With no sentence boundary worth cutting at, the ellipsis says so rather
    # than letting a hard cut read as the end of the text.
    assert backend._tidy("Sentence one. " + "x" * 400).endswith("…")

    assert backend._tidy("") == ""
    assert backend._tidy(None) == ""


def test_a_provider_is_described_by_the_environment_it_reads():
    """There is no prose anywhere in litellm about what a provider *is*, so
    the honest description is the one operational fact it does hold — and it
    is not always an API *key*: Ollama's is a base URL. The sentence has to
    survive both, since local providers are exactly the ones somebody is
    least sure how to configure."""
    backend = _backend()
    made = backend()
    made._litellm = type("L", (), {"validate_environment": staticmethod(
        lambda model: {"keys_in_environment": False,
                       "missing_keys": ["OLLAMA_API_BASE"]})})()

    note = made._provider_note("ollama")

    assert "`OLLAMA_API_BASE`" in note
    assert "key" not in note.lower()


def test_a_model_is_described_only_by_flags_that_say_yes():
    """The ``supports_*`` flags are three-valued in practice — true, false and
    absent — and absent is by far the most common. Reading a missing key as
    "no" would describe most of the map as a model that does nothing, right
    beside the questions asking what this model can read."""
    made = _backend()()

    note = made._model_note({
        "mode": "chat", "litellm_provider": "acme",
        "max_output_tokens": 64000,
        "supports_vision": True, "supports_reasoning": None,
        "supports_function_calling": False,
    })

    assert note == ("A chat model served by acme. Replies are capped at "
                    "64,000 tokens. LiteLLM records support for images.")
    # Nothing known at all is nothing said, not a sentence about an empty map.
    assert made._model_note({}) == ""
