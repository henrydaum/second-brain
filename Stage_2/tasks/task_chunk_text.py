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

from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("ChunkText")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
	"""
	Split text into overlapping chunks, breaking on natural boundaries.

	Tries to split on paragraph breaks, then newlines, then sentence
	endings, then spaces, and finally raw characters as a last resort.
	Each chunk is at most chunk_size characters. Adjacent chunks share
	overlap characters.
	"""
	if not text or not text.strip():
		return []

	if len(text) <= chunk_size:
		return [text]

	# Separators in priority order
	separators = ["\n\n", "\n", ". ", " "]
	chunks = []
	start = 0

	while start < len(text):
		end = start + chunk_size

		if end >= len(text):
			chunks.append(text[start:])
			break

		# Try to find a natural break point within the chunk
		split_at = None
		for sep in separators:
			# Search backwards from the end for a separator
			idx = text.rfind(sep, start, end)
			if idx > start:
				split_at = idx + len(sep)
				break

		if split_at is None:
			# No natural boundary found — hard split at chunk_size
			split_at = end

		chunks.append(text[start:split_at])

		# Move start forward, accounting for overlap
		start = max(start + 1, split_at - overlap)

	return chunks


class ChunkText(BaseTask):
	name = "chunk_text"
	modalities = ["text"]
	reads = ["extracted_text"]
	writes = ["text_chunks"]
	requires_services = []
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
				for i, chunk in enumerate(chunks):
					data.append({
						"path": path,
						"chunk_index": i,
						"content": chunk,
						"char_count": len(chunk),
						"chunked_at": now,
					})

				logger.info(f"Chunked {Path(path).name} into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

				results.append(TaskResult(
					success=True,
					data=data,
				))
			except Exception as e:
				results.append(TaskResult.failed(str(e)))

		return results
