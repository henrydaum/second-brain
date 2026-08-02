"""What the kernel reads off the two tools that move files, either direction.

Same shape as ``test_store_frontend_contracts``: kernel invariants that happen
to be *about* store files. The subject is the kernel's own verdict — does this
load, are these Requests real, does the grant cover what the tool does — and
the store file is the input.

The pair matters together because they are the two directions and they are
easy to confuse. ``read_file`` stages a file for the **model** to look at
(``session.add_attachment``); ``render_files`` hands files back for the
**user** to see (``attachments=`` on the result, which becomes
``ToolResult.attachment_paths``). Neither route reaches the other's
destination.

Skips cleanly when no store ref is reachable.
"""

from pathlib import Path

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source

READ_FILE = "tools/tool_read_file.py"
RENDER_FILES = "tools/tool_render_files.py"
FILE_READS = "tools/helpers/file_reads.py"


def _source_or_skip(relative: str) -> str:
    text = store_source(relative)
    if text is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    return text


def _declarations(relative: str) -> dict:
    from sandbox.validator import validate

    return validate(_source_or_skip(relative),
                    filename=Path(relative).name).declarations


@pytest.mark.parametrize("relative", [READ_FILE, RENDER_FILES, FILE_READS])
def test_the_store_attachment_tools_conform(relative):
    """``conforms`` is the whole question: it means the file loads in a box.

    ``render_files`` was one of the ten store tools still importing
    ``plugins.BaseTool`` and using ``pathlib``, either of which is an ERROR.
    """
    from sandbox.validator import validate

    report = validate(_source_or_skip(relative), filename=Path(relative).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


@pytest.mark.parametrize("relative", [READ_FILE, RENDER_FILES])
def test_every_declared_request_is_a_real_one(relative):
    """``requests`` is the approval grant, so a typo silently narrows it."""
    from guest.requests import ALL_TYPES

    assert set(_declarations(relative)["requests"]) <= set(ALL_TYPES)


def test_read_file_declares_the_staging_request_it_needs():
    """Without it the grant does not cover showing the model a file.

    This is the declaration that made ``read_file`` more than a text reader —
    the kernel opens the path and puts the contents in front of the model, and
    a tool that did not declare it would be refused at the one moment it
    matters.
    """
    declared = _declarations(READ_FILE)
    assert declared["family"] == "tool"
    assert declared["name"] == "read_file"
    assert "session.add_attachment" in declared["requests"]
    # The other half of one door for every file type.
    assert {"parse.modality", "parse.file"} <= set(declared["requests"])
    assert declared["dependencies_files"] == [FILE_READS]


def test_render_files_needs_no_request_to_hand_a_file_back():
    """The user-facing direction rides on the result, not on a Request.

    ``fs.list`` is there to check the paths exist. Nothing else is declared
    because ``sdk.ok(attachments=[...])`` is a *return value* — which is why
    every tool already has this power and only the residual "show the user a
    file nothing produced" case needs a tool of its own.
    """
    declared = _declarations(RENDER_FILES)
    assert declared["family"] == "tool"
    assert declared["name"] == "render_files"
    assert declared["requests"] == ["fs.list"]


def test_neither_tool_is_asked_about():
    """Both are safe for every argument, so neither interrupts a turn.

    ``read_file`` reads and stages, ``render_files`` lists and returns. If
    either ever lands in ``CONSEQUENTIAL`` the agent starts paying a dialog to
    look at its own files, which is the cost that stops it looking.
    """
    from sandbox.policy import CONSEQUENTIAL

    for relative in (READ_FILE, RENDER_FILES):
        declared = _declarations(relative)
        assert not (set(declared["requests"]) & set(CONSEQUENTIAL)), relative
