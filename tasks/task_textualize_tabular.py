"""Turn spreadsheets into markdown tables the model can read.

CSV, Excel, Parquet, SQLite — whatever the tabular parsers handle — rendered as
markdown and stored as text, so the rest of the pipeline treats a spreadsheet
like any other document.

**The DataFrames never leave this box.** ``parse_modalities = ["tabular"]``
provisions the parsers here, ``to_markdown`` runs beside them, and only the
resulting string crosses anything. That is the whole reason tabular is not a
crossable modality: a DataFrame is an intermediate on the way to text, and the
conversion belongs wherever the frame already is.

Capped at ``MAX_ROWS`` per sheet. The point is to let the model reason about
what a spreadsheet *contains*, not to paste a hundred thousand rows into a
context window.
"""

dependencies_files = []
dependencies_pip = []

parse_modalities = ["tabular"]

import time

from guest.bases import BaseTask
from guest.parsing import basename

MAX_ROWS = 50


class TextualizeTabular(BaseTask):
    """Render a file's sheets as markdown."""

    name = "textualize_tabular"
    description = (
        "Render a spreadsheet's sheets as markdown tables the model can "
        "read.")
    modalities = ["tabular"]
    reads = []
    writes = ["tabular_text"]
    requires_services = []
    requests = ["parse.file"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS tabular_text (
            path TEXT PRIMARY KEY,
            content TEXT,
            char_count INTEGER,
            textualized_at REAL
        );
    """
    batch_size = 4
    timeout = 120

    def run(self, sdk, paths):
        """Textualize each path's sheets."""
        now = time.time()
        outcomes = []

        for path in paths:
            try:
                content = self._render(sdk, path)
            except sdk.Failed as failed:
                outcomes.append({"ok": False,
                                 "error": f"parse failed: {failed}"})
                continue

            sdk.log(f"textualized {len(content)} chars from {basename(path)}")
            outcomes.append({
                "ok": True,
                "data": [{
                    "path": path,
                    "content": content,
                    "char_count": len(content),
                    "textualized_at": now,
                }],
            })

        return sdk.ok(per_path=outcomes)

    @staticmethod
    def _render(sdk, path) -> str:
        """Every sheet in one file, as markdown."""
        sheets = sdk.parse.file(path, "tabular") or {}

        sections = []
        for name, frame in sheets.items():
            head = frame.head(MAX_ROWS)
            try:
                table = head.to_markdown(index=False)
            except ImportError:
                # to_markdown needs tabulate. Plain text is a worse table and
                # a perfectly good fallback — losing the sheet entirely
                # because a formatting library is missing would not be.
                table = head.to_string(index=False)

            # A single unnamed sheet needs no heading; several do.
            heading = f"## {name}" if len(sheets) > 1 else ""
            omitted = len(frame) - MAX_ROWS
            footer = f"\n... ({omitted} more rows)" if omitted > 0 else ""

            sections.append("\n".join(part for part in (heading, table, footer)
                                      if part))

        return "\n\n".join(sections)
