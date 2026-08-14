"""What the kernel reads off the store's Telegram frontend, and must keep reading.

These are kernel invariants that happen to be *about* store files. Each one
pins something the kernel itself computes -- the validator's verdict, the
declarations the bridge reads, the isolation the tree resolves -- so the
subject is kernel behaviour and the store file is the input.

The behavioural tests for that frontend (markdown rendering, chunking, the
streamed-reply tracker, media planning) are marked ``store`` and deselected by
default: they exercise code that is not in this repo, and a kernel change
cannot break them. Run them with ``-m store``.

Skips cleanly when no store ref is reachable.
"""

from pathlib import Path

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source

TELEGRAM = "frontends/frontend_telegram.py"
RENDERERS = "frontends/helpers/telegram_renderers.py"


def _source_or_skip(relative: str) -> str:
    text = store_source(relative)
    if text is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    return text


@pytest.fixture(scope="module")
def telegram() -> str:
    return _source_or_skip(TELEGRAM)


# ──────────────────────────────────────────────────────────────────────
# Conformance: the verdict that decides whether a package loads at all.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("relative", [TELEGRAM, RENDERERS])
def test_the_store_frontends_conform(relative):
    """No ERROR findings — warnings about foreign libraries are the point.

    Every file on the store branch now conforms; the unmigrated ones were
    deleted rather than ported, so a store file that fails this is a
    regression rather than a leftover.
    """
    from sandbox.validator import validate

    report = validate(_source_or_skip(relative), filename=Path(relative).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


def test_telegram_declares_what_the_kernel_has_to_know(telegram):
    """The declarations the bridge reads, and cannot infer if they are missing.

    Each of these changes behaviour rather than describing it, and each is
    invisible until it is wrong: without ``background_submit`` the box
    deadlocks against the turn it started, and a ``capabilities`` entry the
    AST cannot evaluate is silently dropped.
    """
    from sandbox.validator import validate

    declared = validate(telegram, filename="frontend_telegram.py").declarations
    assert declared["family"] == "frontend"
    assert declared["name"] == "telegram"
    assert declared["background_submit"] is True
    assert declared["restore_on_start"] is True
    assert declared["dependencies_files"] == [RENDERERS]
    assert "python-telegram-bot" in declared["dependencies_pip"]
    assert declared["capabilities"]["max_upload_size"] == 50 * 1024 * 1024
    assert declared["capabilities"]["max_message_chars"] == 4096


def test_every_declared_request_is_a_real_one(telegram):
    """``requests`` is the approval grant, so a typo silently narrows it."""
    from guest.requests import ALL_TYPES
    from sandbox.validator import validate

    declared = validate(telegram, filename="frontend_telegram.py").declarations
    assert set(declared["requests"]) <= set(ALL_TYPES)


def test_telegram_needs_a_subprocess(telegram):
    """The event loop it owns cannot survive a thread-per-call box.

    An in-process resident box runs each call on a fresh worker thread, so a
    loop created in ``start`` could not be driven from ``poll``. Nothing in the
    file asks for isolation — it is read off the imports, and this pins that
    the reading comes out right.
    """
    from sandbox.validator import validate

    report = validate(telegram, filename="frontend_telegram.py")
    assert "telegram" in report.unmediated
    assert "asyncio" in report.unmediated
