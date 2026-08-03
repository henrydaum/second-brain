"""Scratch files are named after whoever asked for them.

``workspace/temp`` used to be a directory of ``sb-box-<random>``, which threw
away the one fact that makes it readable: opening it and finding unexplained
extracted diagrams is the symptom, and the kernel knew all along that
``extract_container`` had unpacked an archive.

The mechanism is that ``Interpreter._execute`` marks the thread with the
calling execution's chain for the whole handler, so the name is reachable from
inside a handler without being passed. That also reaches code the handler calls
synchronously — which is how the *kernel-side* parser stand-in gets the same
name, despite making no Request at all.
"""

import pytest

from sandbox import provenance
from sandbox.policy import Chain


def _serving(*links, root="agent"):
    """Stand where a handler stands: on a thread marked with a chain."""
    return provenance.serving(Chain(root=root, links=tuple(links)))


# ── The label ─────────────────────────────────────────────────────────

def test_the_prefix_names_the_innermost_caller():
    """The thing that actually asked, not what caused the turn."""
    with _serving("extract_container"):
        assert provenance.scratch_prefix() == "extract_container-"


def test_it_is_the_innermost_link_and_not_the_root():
    """The root is right for a dialog and wrong here: three tasks from one
    turn would all come out identical."""
    with _serving("task_index", "service_web", root="cron:nightly"):
        assert provenance.scratch_prefix() == "service_web-"


def test_a_session_key_cannot_produce_an_illegal_filename():
    """A link can be a session key, and Windows rejects ``:`` outright."""
    with _serving("telegram:7912761600:7912761600:0"):
        prefix = provenance.scratch_prefix()

    assert ":" not in prefix
    assert prefix.startswith("telegram_7912761600")


@pytest.mark.parametrize("link", ["", "   ", "...", "///"])
def test_an_unnameable_caller_still_gets_scratch(link):
    """Falling back matters more than naming: a caller that cannot be
    labelled must not be refused a temp file."""
    with _serving(link):
        assert provenance.scratch_prefix() == "sb-box-"


def test_no_chain_at_all_falls_back():
    """Every test that calls a handler directly stands here."""
    assert provenance.scratch_prefix() == "sb-box-"


def test_a_long_name_is_capped():
    with _serving("x" * 200):
        assert len(provenance.scratch_prefix()) <= 41


# ── Both call sites ───────────────────────────────────────────────────

def test_the_request_handler_uses_it(tmp_path, monkeypatch):
    """``fs.temp``, the route boxed code takes."""
    from tests.support import retarget_trees
    from sandbox.handlers.fs_net import _fs_temp

    retarget_trees(monkeypatch, tmp_path)
    with _serving("extract_container"):
        made = _fs_temp(None, {"directory": True})

    assert made.ok
    assert _name(made.data).startswith("extract_container-")


def test_the_kernel_parser_stand_in_uses_it_too(tmp_path, monkeypatch):
    """The copy that actually filled the folder.

    A parser reached through ``parse.file`` runs *in* the kernel with
    ``KERNEL_SDK``, so it never makes an ``fs.temp`` Request — fixing only the
    handler would have left every extracted archive still called ``sb-box``.
    """
    from tests.support import retarget_trees
    from parsing.kernel_sdk import KERNEL_SDK

    retarget_trees(monkeypatch, tmp_path)
    with _serving("extract_container"):
        made = KERNEL_SDK.fs.temp(directory=True)

    assert _name(made).startswith("extract_container-")


def test_both_sites_agree(tmp_path, monkeypatch):
    """Two implementations of one idea; a test is what keeps them equal."""
    from tests.support import retarget_trees
    from sandbox.handlers.fs_net import _fs_temp
    from parsing.kernel_sdk import KERNEL_SDK

    retarget_trees(monkeypatch, tmp_path)
    with _serving("parse_pdf"):
        handler = _name(_fs_temp(None, {"suffix": ".png"}).data)
        stand_in = _name(KERNEL_SDK.fs.temp(suffix=".png"))

    assert handler.startswith("parse_pdf-") and stand_in.startswith("parse_pdf-")
    assert handler.endswith(".png") and stand_in.endswith(".png")


def _name(path) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]
