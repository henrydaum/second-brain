"""Feed every text a file produced into the FTS5 keyword index.

Four upstreams write text about a path — chunks, OCR, tabular and transcripts —
and this gathers whichever exist into one table. Triggers on that table keep
the FTS5 virtual table in step, so BM25 keyword search covers everything the
pipeline understood about a file, whatever route the text arrived by.

Chunk-level on purpose: BM25 results then line up with embedding results, so
hybrid search can fuse the two rankings without comparing a whole document
against a fragment. The short sources — OCR, tabular, a transcript — get
``chunk_index = 0``.

``modalities`` is empty and ``require_all_inputs`` is False: this is a
downstream task, run when *any* upstream finishes for a path rather than when
a file of some type appears.
"""

dependencies_files = ['tasks/task_chunk_text.py', 'tasks/helpers/rows.py']
dependencies_pip = []

import time

from guest.bases import BaseTask
from guest.parsing import basename

from .rows import paged

#: ``(table, source label, is_chunked)``. Chunked sources keep their own
#: ordering; the rest are one row and take index 0.
SOURCES = (
    ("text_chunks", "extracted", True),
    ("ocr_text", "ocr", False),
    ("tabular_text", "tabular", False),
    ("audio_transcripts", "transcript", False),
)


class IndexLexical(BaseTask):
    """Collect a path's text into the searchable table."""

    name = "index_lexical"
    modalities = []
    reads = ["text_chunks", "ocr_text", "tabular_text", "audio_transcripts"]
    writes = ["lexical_content"]
    require_all_inputs = False
    requires_services = []
    requests = ["db.query"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS lexical_content (
            path TEXT,
            source TEXT,
            chunk_index INTEGER,
            content TEXT,
            char_count INTEGER,
            indexed_at REAL,
            PRIMARY KEY (path, source, chunk_index)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS lexical_index USING fts5(
            path,
            content,
            source,
            chunk_index,
            content=lexical_content,
            content_rowid=rowid,
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS lexical_content_ai AFTER INSERT ON lexical_content BEGIN
            INSERT INTO lexical_index(rowid, path, content, source, chunk_index)
            VALUES (new.rowid, new.path, new.content, new.source, new.chunk_index);
        END;

        CREATE TRIGGER IF NOT EXISTS lexical_content_ad AFTER DELETE ON lexical_content BEGIN
            INSERT INTO lexical_index(lexical_index, rowid, path, content, source, chunk_index)
            VALUES('delete', old.rowid, old.path, old.content, old.source, old.chunk_index);
        END;

        CREATE TRIGGER IF NOT EXISTS lexical_content_au AFTER UPDATE ON lexical_content BEGIN
            INSERT INTO lexical_index(lexical_index, rowid, path, content, source, chunk_index)
            VALUES('delete', old.rowid, old.path, old.content, old.source, old.chunk_index);
            INSERT INTO lexical_index(rowid, path, content, source, chunk_index)
            VALUES (new.rowid, new.path, new.content, new.source, new.chunk_index);
        END;
    """
    batch_size = 8
    timeout = 120

    def run(self, sdk, paths):
        """Gather every available source of text for each path."""
        now = time.time()
        outcomes = []

        for path in paths:
            data = self._collect(sdk, path, now)
            if data:
                found = sorted({row["source"] for row in data})
                sdk.log(f"indexed {len(data)} entries from {basename(path)} "
                        f"({', '.join(found)})")
            outcomes.append({"ok": True, "data": data})

        return sdk.ok(per_path=outcomes)

    @staticmethod
    def _collect(sdk, path, now) -> list:
        """Every row worth indexing for one path.

        A missing upstream is not an error — ``require_all_inputs = False``
        means this runs as soon as *one* of them has produced anything, so
        three of the four being empty is the normal case. The table may not
        even *exist*: ``ocr_text`` is created by ``task_ocr_images``, which is
        a separate package, and querying a table sqlite has never heard of is
        a failed Request rather than an empty answer. Caught per source, so
        one uninstalled package cannot fail the indexing of text that four
        others produced successfully.
        """
        data = []
        for table, label, chunked in SOURCES:
            try:
                # Chunked sources can exceed the 500-row Request cap on a long
                # document, so they are paged; the ORDER BY is what makes that
                # safe, since sqlite need not be consistent between pages.
                if chunked:
                    rows = paged(
                        sdk,
                        f"SELECT chunk_index, content FROM {table} "
                        f"WHERE path = ? ORDER BY chunk_index",
                        [path])
                else:
                    rows = sdk.db.query(
                        f"SELECT content FROM {table} WHERE path = ?",
                        [path])[:1]
            except sdk.Failed as failed:
                sdk.log(f"{table} unavailable for {basename(path)}: {failed}",
                        level="debug")
                continue

            for row in rows:
                content = row.get("content") or ""
                if not content.strip():
                    continue
                data.append({
                    "path": path,
                    "source": label,
                    "chunk_index": row.get("chunk_index", 0) if chunked else 0,
                    "content": content,
                    "char_count": len(content),
                    "indexed_at": now,
                })
        return data
