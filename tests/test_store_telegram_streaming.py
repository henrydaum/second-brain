"""Unit tests for the store Telegram StreamTracker (buffer/throttle/rollover).

The tracker is the pure-logic half of Telegram's streamed-reply rendering
(``frontends/helpers/telegram_renderers.py`` on the ``store`` branch); all
Telegram I/O lives in the frontend's pump, so these tests need neither
python-telegram-bot nor Pillow (a stub PIL is injected for the module's
top-level import).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_STORE_REL = "frontends/helpers/telegram_renderers.py"


def _store_module_source() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_STORE_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def tracker_cls(tmp_path_factory):
    source = _store_module_source()
    if source is None:
        pytest.skip(f"{_STORE_REL} not present on a local store ref")
    if "PIL" not in sys.modules:  # the module imports Pillow at top level
        pil = types.ModuleType("PIL")
        pil.Image = types.ModuleType("PIL.Image")
        sys.modules["PIL"] = pil
    path = tmp_path_factory.mktemp("tg_renderers") / "telegram_renderers.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("store_telegram_renderers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module.StreamTracker


def test_throttle_allows_first_edit_and_blocks_until_interval(tracker_cls):
    t = tracker_cls(edit_interval=2.0, burst_chars=300)
    t.feed("hello")

    assert t.should_edit(now=10.0)  # dirty + never edited
    finals, current = t.take_render()
    assert finals == [] and current == "hello"
    t.mark_rendered(current, now=10.0)

    assert not t.should_edit(now=10.5)  # clean
    t.feed(" world")
    assert not t.should_edit(now=11.0)  # dirty but inside the interval
    assert t.should_edit(now=12.5)      # interval elapsed


def test_burst_chars_bypass_the_interval(tracker_cls):
    t = tracker_cls(edit_interval=60.0, burst_chars=10)
    t.feed("x")
    _, current = t.take_render()
    t.mark_rendered(current, now=10.0)

    t.feed("y" * 20)
    assert t.should_edit(now=10.1)  # 20 new chars >= burst threshold


def test_unconfirmed_render_is_retried(tracker_cls):
    t = tracker_cls()
    t.feed("abc")

    _, first = t.take_render()
    assert first == "abc"
    # Edit failed (RetryAfter): not marked — the same text is offered again.
    _, second = t.take_render()
    assert second == "abc"
    t.mark_rendered(second, now=1.0)
    _, third = t.take_render()
    assert third is None  # confirmed → clean


def test_rollover_splits_on_newline_and_sets_rolled(tracker_cls):
    t = tracker_cls(max_chars=100)
    head = "a" * 80
    tail = "b" * 60
    t.feed(head + "\n" + tail)

    finals, current = t.take_render()

    assert finals == [head]
    assert current == tail
    assert t.rolled is True


def test_rollover_hard_splits_without_newline(tracker_cls):
    t = tracker_cls(max_chars=100)
    t.feed("c" * 250)

    finals, current = t.take_render()

    assert finals == ["c" * 100, "c" * 100]
    assert current == "c" * 50
    assert t.rolled


def test_finish_and_state_round_trip(tracker_cls):
    t = tracker_cls()
    t.feed("partial")
    assert t.state() == (False, False, None)

    t.finish("Final text.", aborted=False)
    assert t.state() == (True, False, "Final text.")
    assert t.remainder() == "partial"

    t2 = tracker_cls()
    t2.finish(None, aborted=True)
    assert t2.state() == (True, True, None)
