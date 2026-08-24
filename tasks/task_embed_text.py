"""Embed a batch of files' text chunks in one call to the model.

Chunks are pooled across every file in the batch and encoded together, because
that is what a GPU is good at: one call over four hundred chunks costs a
fraction of four hundred calls over one, and the service's box serializes its
calls anyway, so per-file encoding buys no parallelism to trade for it.

Vectors come back as raw float32 buffers — one ``bytes`` per chunk, exactly
what the ``embedding`` BLOB column stores. Bytes cross the boundary natively,
so nothing here encodes or decodes anything; the value handed to the row is
the value sqlite writes.
"""

dependencies_files = ['services/service_embed.py', 'tasks/task_chunk_text.py',
                      'tasks/helpers/rows.py']
dependencies_pip = []

import time

from guest.bases import BaseTask
from guest.parsing import basename

from .rows import paged


class EmbedText(BaseTask):
    """One vector per text chunk."""

    name = "embed_text"
    description = (
        "Embed a file's text chunks into vectors for semantic search, a "
        "batch of files per model call.")
    modalities = ["text"]
    reads = ["text_chunks"]
    writes = ["text_embeddings"]
    requires_services = ["text_embedder"]
    requests = ["db.query", "service.call"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS text_embeddings (
            path TEXT,
            chunk_index INTEGER,
            embedding BLOB,
            model_name TEXT,
            embedded_at REAL,
            PRIMARY KEY (path, chunk_index)
        );
    """
    batch_size = 4
    max_workers = 4
    timeout = 300

    def run(self, sdk, paths):
        """Pool the batch's chunks, encode once, hand each file its slice."""
        described = sdk.services.call("text_embedder", "describe") or {}
        # The HF model id, not the service name. It is stored on every row and
        # matched in ``WHERE model_name = ?`` at search time, so reading it off
        # the adapter — which answers "text_embedder" — would make every vector
        # unfindable by the model that produced it.
        model_name = described.get("model_name") or "unknown"
        now = time.time()

        # ── 1. Gather, per file ────────────────────────────────────────
        chunks = {}          # path -> [(chunk_index, content), ...]
        failures = {}        # path -> why
        for path in paths:
            try:
                rows = paged(
                    sdk,
                    "SELECT chunk_index, content FROM text_chunks "
                    "WHERE path = ? ORDER BY chunk_index",
                    [path])
            except sdk.Failed as failed:
                failures[path] = f"could not read chunks: {failed}"
                continue
            chunks[path] = [(row["chunk_index"], row["content"] or "")
                            for row in rows]

        pooled = [content for path in paths
                  for _index, content in chunks.get(path, ())]

        # ── 2. One encode for the whole batch ──────────────────────────
        vectors = []
        if pooled:
            try:
                sdk.log(f"encoding {len(pooled)} chunk(s) across "
                        f"{len(chunks)} file(s)")
                vectors = sdk.services.call("text_embedder", "encode",
                                            inputs=pooled) or []
            except sdk.Failed as failed:
                return sdk.ok(per_path=[{"ok": False,
                                         "error": f"encode failed: {failed}"}
                                        for _ in paths])

            if len(vectors) != len(pooled):
                # Silence here would write vectors against the wrong chunks:
                # the slicing below is positional, so a short answer shifts
                # every file after the first by an unknown amount.
                return sdk.ok(per_path=[
                    {"ok": False,
                     "error": f"embedder returned {len(vectors)} vectors for "
                              f"{len(pooled)} chunks"} for _ in paths])

        # ── 3. Slice it back out, in the order it went in ──────────────
        outcomes = []
        cursor = 0
        for path in paths:
            if path in failures:
                outcomes.append({"ok": False, "error": failures[path]})
                continue

            mine = chunks.get(path, [])
            rows = [{
                "path": path,
                "chunk_index": chunk_index,
                "embedding": vectors[cursor + offset],
                "model_name": model_name,
                "embedded_at": now,
            } for offset, (chunk_index, _content) in enumerate(mine)]
            cursor += len(mine)

            if rows:
                sdk.log(f"embedded {len(rows)} chunk(s) from {basename(path)}",
                        level="debug")
            outcomes.append({"ok": True, "data": rows})

        return sdk.ok(per_path=outcomes)
