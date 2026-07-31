"""Split extracted text into overlapping chunks for embedding.

Character-based, breaking on natural boundaries: paragraphs, then newlines,
then sentences, then words, then characters. Overlap carries the tail of one
chunk into the next so meaning is not lost at a boundary — which is what makes
the embeddings downstream worth anything.

Pure stdlib, which is worth noticing: with nothing foreign in it this task runs
*in-process* even from the installed tree, so chunking a batch costs no process
spawn. The three functions below moved across from the native version
unchanged; only ``run`` had to be rewritten.
"""

dependencies_files = ['tasks/task_extract_text.py']
dependencies_pip = []

import time

from guest.bases import BaseTask
from guest.parsing import basename


def _looks_like_gibberish(text: str) -> bool:
    """Conservative reject: only flag chunks almost certainly junk.

    Mojibake, binary leakage, replacement-character soup — the output of a
    parser that failed without saying so. The bias is deliberately toward
    keeping content: a false negative wastes a little index space, a false
    positive silently loses the user's data. Two cheap signals:

      1. Replacement-char density > 5% — the U+FFFD pattern from bad decodes.
      2. Word-like ratio < 25% over 80+ non-space chars — binary noise, hex
         dumps, control-character streams. The 80-char floor keeps short
         legitimate fragments (numbers, code, table cells) safe.
    """
    if not text:
        return False

    if text.count("�") / max(len(text), 1) > 0.05:
        return True

    non_space = [c for c in text if not c.isspace()]
    if len(non_space) < 80:
        return False

    wordlike = sum(1 for c in non_space if c.isalpha() or c.isdigit())
    return (wordlike / len(non_space)) < 0.25


def _recursive_split(text: str, separators: list, chunk_size: int) -> list:
    """Break text into atomic segments, coarsest boundary first.

    Try paragraphs; anything still too large recurses onto the next finer
    separator. This preserves natural reading boundaries as far as possible,
    which is the difference between a chunk that means something and one that
    starts mid-word.
    """
    if not text:
        return []

    separator = separators[0]
    remaining = separators[1:]

    # Empty separator is the character-level floor: the text is already the
    # smallest unit that can be produced.
    if not separator:
        return [text]

    splits = text.split(separator)

    segments = []
    for index, piece in enumerate(splits):
        # Re-attach the separator so whitespace survives the round trip.
        if index < len(splits) - 1:
            piece += separator
        if not piece:
            continue
        if len(piece) <= chunk_size or not remaining:
            segments.append(piece)
        else:
            segments.extend(_recursive_split(piece, remaining, chunk_size))

    return segments


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Split into overlapping chunks that break on natural boundaries."""
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    segments = _recursive_split(text, separators, chunk_size)

    chunks = []
    current = []
    current_len = 0

    for segment in segments:
        length = len(segment)

        # A single "word" longer than a chunk. Rare, and unsplittable — emit
        # it alone rather than letting it push a real chunk out of shape.
        if length > chunk_size:
            if current:
                chunks.append("".join(current))
                current = []
                current_len = 0
            chunks.append(segment)
            continue

        if current_len + length > chunk_size:
            chunks.append("".join(current))

            # Carry the tail of the finished chunk into the next one, walking
            # backwards until roughly ``overlap`` characters are accumulated.
            carried = []
            carried_len = 0
            for previous in reversed(current):
                if carried_len + len(previous) > overlap:
                    break
                carried.insert(0, previous)
                carried_len += len(previous)

            current = carried
            current_len = carried_len

        current.append(segment)
        current_len += length

    if current:
        chunks.append("".join(current))

    return chunks


class ChunkText(BaseTask):
    """Turn one file's extracted text into indexable chunks."""

    name = "chunk_text"
    modalities = ["text"]
    reads = ["extracted_text"]
    writes = ["text_chunks"]
    requires_services = []
    requests = ["config.read", "db.query"]
    config_settings = [
        # Both were read by the native task and neither was declared, so
        # ``/config`` could not show or change either — the same
        # undeclared-setting bug service_whisper had.
        ("Chunk Size", "embed_chunk_size",
         "Target chunk length in characters. Larger chunks carry more context "
         "per embedding and retrieve less precisely.",
         512,
         {"type": "slider", "range": (128, 2048, 128), "is_float": False}),

        ("Chunk Overlap", "embed_chunk_overlap",
         "Characters carried from the end of one chunk into the start of the "
         "next, so meaning is not lost at a boundary.",
         50,
         {"type": "slider", "range": (0, 200, 40), "is_float": False}),
    ]
    output_schema = """
        CREATE TABLE IF NOT EXISTS text_chunks (
            path TEXT,
            chunk_index INTEGER,
            content TEXT,
            char_count INTEGER,
            chunked_at REAL,
            PRIMARY KEY (path, chunk_index)
        );
    """
    batch_size = 8
    timeout = 120

    def run(self, sdk, paths):
        """Chunk each path's extracted text."""
        chunk_size = int(sdk.config.read("embed_chunk_size") or 512)
        overlap = int(sdk.config.read("embed_chunk_overlap") or 50)
        now = time.time()
        outcomes = []

        for path in paths:
            try:
                rows = sdk.db.query(
                    "SELECT content FROM extracted_text WHERE path = ?",
                    [path])
            except sdk.Failed as failed:
                outcomes.append({"ok": False, "error": str(failed)})
                continue

            if not rows:
                outcomes.append({"ok": False,
                                 "error": "no extracted text found"})
                continue

            content = rows[0].get("content") or ""
            if not content.strip():
                # Nothing to chunk is a success with no rows, not a failure:
                # an empty file is extracted correctly and has no chunks.
                outcomes.append({"ok": True, "data": []})
                continue

            chunks = _chunk_text(content, chunk_size, overlap)

            data = []
            dropped = 0
            for chunk in chunks:
                if _looks_like_gibberish(chunk):
                    dropped += 1
                    continue
                # Indexed by what was *kept*, so the sequence has no holes.
                data.append({
                    "path": path,
                    "chunk_index": len(data),
                    "content": chunk,
                    "char_count": len(chunk),
                    "chunked_at": now,
                })

            sdk.log(f"chunked {basename(path)} into {len(data)} chunks"
                    + (f" (dropped {dropped} as gibberish)" if dropped else "")
                    + f" (size={chunk_size}, overlap={overlap})")
            outcomes.append({"ok": True, "data": data})

        return sdk.ok(per_path=outcomes)
