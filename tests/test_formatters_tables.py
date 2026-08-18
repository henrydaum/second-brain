"""Tests for the shared markdown-table primitives in formatters.py.

Commands emit GitHub-style markdown tables; monospace surfaces (REPL,
Telegram's <pre> fallback) align them with align_md_tables while rich
surfaces render the markdown natively.
"""

import re

from bundled.frontends.helpers.formatters import align_md_tables, md_table, render_plain
from plugins.command_registry import CommandRegistry
from plugins.native.command import BaseCommand


def _tables_start_their_own_block(text: str) -> bool:
    """Every table must be preceded by a blank line, or GFM parsers fold it
    into the previous paragraph and render the pipes inline."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        starts_table = line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*-{3,}\s*\|)+$", lines[i + 1].replace(" ", " "))
        if starts_table and i > 0 and lines[i - 1].strip() != "":
            return False
    return True


def test_md_table_shape_and_escaping():
    table = md_table(["Name", "Count"], [("a|b", 1), ("plain", None)])
    lines = table.split("\n")

    assert lines[0] == "| Name | Count |"
    assert set(lines[1]) <= {"|", "-", " "}
    assert "a\\|b" in lines[2]
    assert lines[3] == "| plain |  |"


def test_align_md_tables_pads_columns():
    table = md_table(["Category", "Count"], [("Tools", 16), ("Frontends", 2)])
    aligned = align_md_tables(table)
    lines = aligned.split("\n")

    assert "|" not in aligned
    assert lines[0].startswith("Category")
    assert lines[2].startswith("Tools")
    # Every count sits in the same column.
    assert lines[2].index("16") == lines[3].index("2")


def test_align_md_tables_round_trips_escaped_pipes():
    aligned = align_md_tables(md_table(["X"], [("a|b",)]))
    assert "a|b" in aligned
    assert "\\|" not in aligned


def test_render_plain_strips_fence_markers_and_aligns():
    text = "**State**\n```\nTurn: user\nPhase: base\n```\n\n" + md_table(["A", "B"], [(1, 2)])
    out = render_plain(text)

    assert "```" not in out
    assert "Turn: user\nPhase: base" in out
    assert "| 1 | 2 |" not in out  # table got aligned too


def test_align_md_tables_leaves_prose_untouched():
    text = "Header text.\n\nNo table here | just a stray pipe.\n"
    assert align_md_tables(text) == text


def test_command_help_tables_start_their_own_block():
    """help_text() is the one live md_table caller, and it interleaves tables
    with **bold** category headings -- exactly the shape that folds."""
    class _Cmd(BaseCommand):
        name = "demo"
        description = "A demo command"
        category = "System"

    registry = CommandRegistry()
    registry.register(_Cmd())
    text = registry.help_text()

    assert _tables_start_their_own_block(text)
    assert "**System**\n\n|" in text


def test_align_md_tables_handles_table_between_prose():
    text = "Installed files by category:\n\n" + md_table(["A", "B"], [(1, 2)]) + "\n\nChoose a category."
    aligned = align_md_tables(text)

    assert aligned.startswith("Installed files by category:")
    assert aligned.endswith("Choose a category.")
    assert "| 1 | 2 |" not in aligned
    assert "1  2" in aligned
