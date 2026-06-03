"""
Chunk Text task.

Reads extracted text from the extracted_text table and splits it into
overlapping chunks for downstream embedding. Uses character-based splitting
with natural boundary detection (paragraphs → newlines → sentences → words).

Config keys:
    embed_chunk_size    Target chunk size in characters (default 512)
    embed_chunk_overlap Overlap between adjacent chunks (default 50)
"""

import logging
import time
from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("ChunkText")


def _looks_like_gibberish(text: str) -> bool:
	"""Conservative reject: only flag chunks that are almost certainly junk
	from a broken parser (mojibake, binary leakage, replacement-char soup).

	Bias is toward keeping content — false-negative is fine, false-positive
	(dropping real text) is not. Two cheap signals:
	  1. Replacement-char density > 5% — the U+FFFD "�" pattern from bad decodes.
	  2. Alpha-or-CJK ratio < 25% over a chunk of 80+ non-space chars — catches
	     binary noise, hex dumps, control-char streams. The 80-char floor
	     keeps short legit fragments (numbers, code, tables) safe.
	"""
	if not text:
		return False

	# Replacement chars: very strong signal, low false-positive rate.
	if text.count("�") / max(len(text), 1) > 0.05:
		return True

	# Skip the alpha-ratio check on short chunks — too easy to false-positive
	# on legitimate numeric/code/table fragments.
	non_space = [c for c in text if not c.isspace()]
	if len(non_space) < 80:
		return False

	def _is_wordlike(c: str) -> bool:
		# Letters in any script (Latin, CJK, Cyrillic, Arabic, etc.) plus digits.
		"""Return whether wordlike."""
		return c.isalpha() or c.isdigit()

	wordlike = sum(1 for c in non_space if _is_wordlike(c))
	return (wordlike / len(non_space)) < 0.25


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
	"""
	Break text into atomic segments by trying progressively finer separators.

	Strategy: Try splitting on the coarsest boundary first (paragraphs).
	If any piece is still too large, recurse with the next finer separator
	(newlines -> sentences -> words -> characters). This preserves natural
	reading boundaries as much as possible.
	"""
	if not text:
		return []

	sep = separators[0]
	remaining_seps = separators[1:]

	# Base case: empty separator means single-character splitting.
	# The text itself is the smallest atomic unit we can produce.
	if not sep:
		return [text]

	splits = text.split(sep)

	segments = []
	for i, s in enumerate(splits):
		# Re-attach separator to preserve whitespace in output (except last piece)
		if i < len(splits) - 1:
			s += sep
		if not s:
			continue

		# If this piece fits in a chunk, keep it as-is.
		# Otherwise, recurse with a finer separator to break it down further.
		if len(s) <= chunk_size or not remaining_seps:
			segments.append(s)
		else:
			segments.extend(_recursive_split(s, remaining_seps, chunk_size))

	return segments


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
	"""
	Split text into overlapping chunks, breaking on natural boundaries.

	Two-phase approach:
	1. Recursively split text into atomic segments using natural boundaries
	   (paragraphs → newlines → sentences → words → characters).
	2. Merge segments into chunks up to chunk_size, with overlap between
	   adjacent chunks.
	"""
	if not text or not text.strip():
		return []

	if len(text) <= chunk_size:
		return [text]

	# Separator hierarchy: try coarse boundaries first, fall back to finer ones.
	# Empty string at the end is the "character-level" fallback.
	separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
	segments = _recursive_split(text, separators, chunk_size)

	chunks = []
	current_chunk = []
	current_len = 0

	for segment in segments:
		seg_len = len(segment)

		# Oversized segment that couldn't be split further — emit as-is.
		# This only happens when a single "word" exceeds chunk_size (rare).
		if seg_len > chunk_size:
			if current_chunk:
				chunks.append("".join(current_chunk))
				current_chunk = []
				current_len = 0
			chunks.append(segment)
			logger.debug(f"Oversized segment ({seg_len} chars) emitted as standalone chunk")
			continue

		# Adding this segment would exceed chunk_size — finalize current chunk
		if current_len + seg_len > chunk_size:
			chunks.append("".join(current_chunk))

			# Overlap: carry the tail of the previous chunk into the next one.
			# This ensures context isn't lost at chunk boundaries — critical for
			# embedding quality. Walk backwards through segments until we've
			# accumulated ~overlap characters.
			overlap_buffer = []
			overlap_len = 0
			for prev_seg in reversed(current_chunk):
				prev_len = len(prev_seg)
				if overlap_len + prev_len > overlap:
					break
				overlap_buffer.insert(0, prev_seg)
				overlap_len += prev_len

			current_chunk = overlap_buffer
			current_len = overlap_len

		current_chunk.append(segment)
		current_len += seg_len

	if current_chunk:
		chunks.append("".join(current_chunk))

	return chunks


class ChunkText(BaseTask):
	"""Chunk text."""
	name = "chunk_text"
	modalities = ["text"]
	reads = ["extracted_text"]
	writes = ["text_chunks"]
	requires_services = []
	config_settings = [
		("Chunk Overlap", "embed_chunk_overlap",
		 "Number of overlapping tokens between chunks. Preserves continuity across chunk boundaries.",
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

	def run(self, paths, context):
		"""Run chunk text."""
		chunk_size = context.config.get("embed_chunk_size", 512)
		overlap = context.config.get("embed_chunk_overlap", 50)
		now = time.time()
		results = []

		for path in paths:
			try:
				# Read extracted text from upstream task's output table
				rows = context.db.get_task_output("extracted_text", path)
				if not rows:
					results.append(TaskResult.failed("No extracted text found"))
					continue

				content = rows[0]["content"] or ""
				if not content.strip():
					results.append(TaskResult(
						success=True,
						data=[],
					))
					continue

				chunks = _chunk_text(content, chunk_size, overlap)

				data = []
				dropped = 0
				kept_index = 0
				for chunk in chunks:
					if _looks_like_gibberish(chunk):
						dropped += 1
						continue
					data.append({
						"path": path,
						"chunk_index": kept_index,
						"content": chunk,
						"char_count": len(chunk),
						"chunked_at": now,
					})
					kept_index += 1

				if dropped:
					logger.info(
						f"Chunked {Path(path).name} into {len(chunks)} chunks; "
						f"dropped {dropped} as gibberish (size={chunk_size}, overlap={overlap})"
					)
				else:
					logger.info(f"Chunked {Path(path).name} into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

				results.append(TaskResult(
					success=True,
					data=data,
				))
			except Exception as e:
				results.append(TaskResult.failed(str(e)))

		return results
