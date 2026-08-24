"""Extract text from a file and store it.

The foundation of the pipeline: chunking, embedding and lexical indexing all
read what this writes. It is also the one task in the bundle that needs no
parser of its own, because text is *crossable* — the kernel routes to whichever
parser handles the extension, runs it wherever that parser belongs, and answers
with a string. Declaring ``parse_modalities`` here would drag PyMuPDF,
python-docx and the rest into this box for nothing.

``also_contains`` is why a PDF full of charts ever reaches the image tasks: the
parser reports what *else* was in the file and the orchestrator re-enqueues the
path into every task claiming that modality. It rides on the per-path entry,
because it is a fact about one file rather than about the batch.
"""

dependencies_files = []
dependencies_pip = []

import time

from guest.bases import BaseTask
from guest.parsing import basename


class ExtractText(BaseTask):
    """Parse a file to text and record it."""

    name = "extract_text"
    description = (
        "Parse a file to plain text using the installed parser for its "
        "type. The first step of the path pipeline; everything else "
        "reads what it writes.")
    modalities = ["text"]
    reads = []
    writes = ["extracted_text"]
    requires_services = []
    requests = ["parse.file"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS extracted_text (
            path TEXT PRIMARY KEY,
            content TEXT,
            char_count INTEGER,
            also_contains TEXT,
            extracted_at REAL
        );
    """
    batch_size = 8
    timeout = 120

    def run(self, sdk, paths):
        """Extract each path, reporting them one at a time.

        Per-path rather than one verdict for the batch: an unreadable file
        must not fail the seven beside it, and the orchestrator marks each
        path from its own entry.
        """
        now = time.time()
        outcomes = []

        for path in paths:
            try:
                parsed = sdk.parse.file(path, "text", detail=True)
            except sdk.Failed as failed:
                outcomes.append({"ok": False,
                                 "error": f"parse failed: {failed}"})
                continue

            content = parsed.get("output") or ""
            nested = list(parsed.get("also_contains") or [])
            sdk.log(f"extracted {len(content)} chars from {basename(path)}"
                    + (f", also contains {nested}" if nested else ""))

            outcomes.append({
                "ok": True,
                "also_contains": nested,
                "data": [{
                    "path": path,
                    "content": content,
                    "char_count": len(content),
                    "also_contains": ",".join(nested),
                    "extracted_at": now,
                }],
            })

        return sdk.ok(per_path=outcomes)
