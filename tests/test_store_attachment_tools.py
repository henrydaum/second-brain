"""What the kernel reads off the two tools that move files, either direction.

Same shape as ``test_store_frontend_contracts``: kernel invariants that happen
to be *about* store files. The subject is the kernel's own verdict — does this
load, are these Requests real, does the grant cover what the tool does — and
the store file is the input.

The pair matters together because they are the two directions and they are
easy to confuse. ``read_file`` stages a file for the **model** to look at
(``session.add_attachment``); ``show_files`` hands files back for the
**user** to see (``attachments=`` on the result, which becomes
``ToolResult.attachment_paths``). Neither route reaches the other's
destination.

Skips cleanly when no store ref is reachable.
"""

import ast
from pathlib import Path

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source

READ_FILE = "tools/tool_read_file.py"
SHOW_FILES = "tools/tool_show_files.py"
FILE_READS = "tools/helpers/file_reads.py"
PATH_REPAIR = "tools/helpers/path_repair.py"


def _source_or_skip(relative: str) -> str:
    text = store_source(relative)
    if text is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    return text


def _declarations(relative: str) -> dict:
    from sandbox.validator import validate

    return validate(_source_or_skip(relative),
                    filename=Path(relative).name).declarations


@pytest.mark.parametrize("relative", [READ_FILE, SHOW_FILES, FILE_READS])
def test_the_store_attachment_tools_conform(relative):
    """``conforms`` is the whole question: it means the file loads in a box.

    ``show_files`` was one of the ten store tools still importing
    ``plugins.BaseTool`` and using ``pathlib``, either of which is an ERROR.
    """
    from sandbox.validator import validate

    report = validate(_source_or_skip(relative), filename=Path(relative).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


@pytest.mark.parametrize("relative", [READ_FILE, SHOW_FILES])
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
    # Membership rather than an exact list. Pinning the whole list made
    # every new shared helper a test failure in a file about *staging*,
    # which says nothing about whether the helper belongs there.
    assert FILE_READS in declared["dependencies_files"]
    assert PATH_REPAIR in declared["dependencies_files"]


def test_show_files_needs_no_request_to_hand_a_file_back():
    """The user-facing direction rides on the result, not on a Request.

    ``fs.list`` is there to check the paths exist. Nothing else is declared
    because ``sdk.ok(attachments=[...])`` is a *return value* — which is why
    every tool already has this power and only the residual "show the user a
    file nothing produced" case needs a tool of its own.
    """
    declared = _declarations(SHOW_FILES)
    assert declared["family"] == "tool"
    assert declared["name"] == "show_files"
    assert declared["requests"] == ["fs.list"]


def test_neither_tool_is_asked_about():
    """Both are safe for every argument, so neither interrupts a turn.

    ``read_file`` reads and stages, ``show_files`` lists and returns. If
    either ever lands in ``CONSEQUENTIAL`` the agent starts paying a dialog to
    look at its own files, which is the cost that stops it looking.
    """
    from sandbox.policy import CONSEQUENTIAL

    for relative in (READ_FILE, SHOW_FILES):
        declared = _declarations(relative)
        assert not (set(declared["requests"]) & set(CONSEQUENTIAL)), relative


# ────────────────────────────────────────────────────────────────────
# Routing: which branch a file takes, and in what order.
# ────────────────────────────────────────────────────────────────────

def _run_body() -> ast.FunctionDef:
    """The tool's ``run`` method, as an AST node."""
    tree = ast.parse(_source_or_skip(READ_FILE))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    raise AssertionError("tool_read_file has no run()")


def test_read_file_asks_which_parser_owns_the_extension():
    """The bare call cannot answer it: parse_text registers .py as "text" and
    parse_gdoc registers .gdoc as "text", so only ``detail=True`` separates a
    source file from a pointer to a document living in Drive."""
    body = _run_body()

    detailed = [
        node for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "modality"
        and any(kw.arg == "detail" for kw in node.keywords)
    ]

    assert detailed, (
        "read_file resolves the extension without asking whether the parser "
        "behind it is the generic text fallback — the .gdoc bug")


def test_the_parser_branch_runs_before_any_bytes_are_read():
    """The whole bug was branch *order*, not a missing branch.

    A .gdoc is a small JSON stub: it decodes cleanly, so the binary sniff that
    used to be the only route to a parser said "this is text" and the parser
    was never consulted. Deciding from the registry has to happen before
    ``fs.read``, or the sniff gets there first and answers wrongly again.
    """
    body = _run_body()

    first_parse = first_read = None
    for node in ast.walk(body):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "_parse":
            if first_parse is None or node.lineno < first_parse:
                first_parse = node.lineno
        if (isinstance(func, ast.Attribute) and func.attr == "read"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "fs"):
            if first_read is None or node.lineno < first_read:
                first_read = node.lineno

    assert first_parse is not None and first_read is not None
    assert first_parse < first_read, (
        "the first route to a parser is still behind fs.read, so a file whose "
        "text is not its content is read as though it were")


def test_plain_text_still_reaches_fs_read_for_edit_files_sake():
    """Routing everything through the parser would be the obvious fix and a
    wrong one: parse_text applies clean_text and a char cap, and edit_file's
    exact-replacement gate needs what is byte-for-byte on disk. So the generic
    route must stay on fs.read, and the tool must say why."""
    source = _source_or_skip(READ_FILE)
    body = _run_body()

    guarded = [
        node for node in ast.walk(body)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "generic"
    ]

    assert guarded, "nothing in run() consults the generic flag"
    assert "edit_file" in source, (
        "the reason plain text stays on fs.read is not written down, which is "
        "how it gets 'simplified' into the parser branch later")

