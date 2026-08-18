"""Markdown-table primitives for monospace surfaces.

Two entry points, and they are not a matched pair. ``md_table`` builds one
GitHub-style table; ``render_plain`` takes a whole message body and renders it
for a terminal -- aligning any tables in it and dropping code-fence markers.

**This module is almost all gone, and what happened to it is worth knowing
before adding to it.** It used to be the presentation library for the app --
badges, detail cards, quote blocks, and a ``format_*`` per subsystem -- because
commands and frontends were native code that could import it. They are
sandboxed guests now, and a guest cannot import a host module at all. The live
versions of every one of those primitives are ``sdk.md.*`` in
``sandbox/guest/sdk.py``; that is where a command or frontend reaches for one,
and where a new one goes.

What is left is what still has a caller on *this* side of the boundary:
``md_table`` for ``plugins/command_registry.py``, which builds the command
catalog before any guest is involved, and ``render_plain`` as the oracle
``tests/test_sandbox_console.py`` pins ``sdk.md.plain`` against. The two
implementations are deliberately not shared code -- the guest half is
stdlib-only and self-contained -- and a test is what keeps them agreeing.
"""

import re


def md_table(headers: list, rows: list) -> str:
    """Build a GitHub-style markdown table from headers and row tuples."""
    def cell(value) -> str:
        return str("" if value is None else value).replace("\n", " ").replace("|", "\\|")
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)

def render_plain(text: str) -> str:
    """Render markdown-ish output for a monospace terminal: align tables
    and drop code-fence markers (the content already reads as plain text)."""
    aligned = align_md_tables(text)
    return "\n".join(line for line in aligned.split("\n") if not re.fullmatch(r"\s*```\w*\s*", line))

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

_TABLE_SEPARATOR = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")

def _split_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]

def align_md_tables(text: str) -> str:
    """Render markdown tables in *text* as padded monospace columns.

    Non-table lines pass through untouched, so the same message body works
    on rich and plain surfaces alike.
    """
    lines = (text or "").split("\n")
    out, i = [], 0
    while i < len(lines):
        if (_TABLE_ROW.match(lines[i]) and i + 1 < len(lines)
                and _TABLE_SEPARATOR.match(lines[i + 1])):
            block = [lines[i]]
            j = i + 2
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                block.append(lines[j])
                j += 1
            rows = [_split_row(line) for line in block]
            n = max(len(r) for r in rows)
            rows = [r + [""] * (n - len(r)) for r in rows]
            widths = [max(len(r[c]) for r in rows) for c in range(n)]
            def fmt(row):
                return "  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip()
            out.append(fmt(rows[0]))
            out.append("  ".join("-" * w for w in widths))
            out.extend(fmt(r) for r in rows[1:])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)
