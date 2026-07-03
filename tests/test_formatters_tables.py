"""Tests for the shared markdown-table primitives in formatters.py.

Commands emit GitHub-style markdown tables; monospace surfaces (REPL,
Telegram's <pre> fallback) align them with align_md_tables while rich
surfaces render the markdown natively.
"""

import re

from plugins.frontends.helpers.formatters import align_md_tables, detail_card, format_tasks, md_table


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


def test_detail_card_title_becomes_header():
    card = detail_card("default (active)", [("LLM", "default"), ("Tool list", "(none)")])
    aligned = align_md_tables(card)
    lines = aligned.split("\n")

    assert card.startswith("| default (active) |  |")
    assert lines[0] == "default (active)"
    assert lines[2].startswith("LLM")
    assert "(none)" in lines[3]


def test_align_md_tables_leaves_prose_untouched():
    text = "Header text.\n\nNo table here | just a stray pipe.\n"
    assert align_md_tables(text) == text


def test_task_section_tables_start_their_own_block():
    text = format_tasks([
        {"name": "index", "trigger": "path", "counts": {}, "paused": False},
        {"name": "spawn", "trigger": "event", "counts": {}, "paused": True},
    ])
    assert _tables_start_their_own_block(text)
    assert "**Path-driven tasks**\n\n|" in text


def test_align_md_tables_handles_table_between_prose():
    text = "Installed files by category:\n\n" + md_table(["A", "B"], [(1, 2)]) + "\n\nChoose a category."
    aligned = align_md_tables(text)

    assert aligned.startswith("Installed files by category:")
    assert aligned.endswith("Choose a category.")
    assert "| 1 | 2 |" not in aligned
    assert "1  2" in aligned
