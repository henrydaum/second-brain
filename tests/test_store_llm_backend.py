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
