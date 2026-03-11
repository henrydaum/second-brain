"""
Index Search task.

Reads text chunks and/or OCR text and writes them to the search_content
table, which feeds an FTS5 full-text search index via SQLite triggers.
Enables BM25-ranked keyword search across all indexed files.

Indexes at chunk level so BM25 results align with embedding results
for hybrid search fusion. OCR text (typically short) gets chunk_index=0.

This is a downstream task — no modalities needed. It runs whenever
either upstream (chunk_text or ocr_images) completes for a path.

Depends on text_chunks (produced by chunk_text) OR ocr_text (produced
by ocr_images). require_all_inputs = False means either suffices.
"""

import logging
import time
from pathlib import Path

from Stage_2.BaseTask import BaseTask, TaskResult
from Stage_2.tasks.search_schema import SEARCH_SCHEMA

logger = logging.getLogger("IndexSearch")


class IndexSearch(BaseTask):
	name = "index_search"
	modalities = []  # downstream task — triggered by upstream completion
	reads = ["text_chunks", "ocr_text"]
	writes = ["search_content"]
	require_all_inputs = False  # run when either input exists
	requires_services = []
	output_schema = SEARCH_SCHEMA
	batch_size = 8
	timeout = 120

	def run(self, paths, context):
		now = time.time()
		results = []

		for path in paths:
			try:
				data = []

				# Index text chunks if available
				chunks = context.db.get_task_output("text_chunks", path)
				if chunks:
					for row in chunks:
						content = row["content"] or ""
						if content.strip():
							data.append({
								"path": path,
								"source": "extracted",
								"chunk_index": row["chunk_index"],
								"content": content,
								"char_count": len(content),
								"indexed_at": now,
							})

				# Index OCR text if available
				ocr = context.db.get_task_output("ocr_text", path)
				if ocr:
					content = ocr[0]["content"] or ""
					if content.strip():
						data.append({
							"path": path,
							"source": "ocr",
							"chunk_index": 0,
							"content": content,
							"char_count": len(content),
							"indexed_at": now,
						})

				results.append(TaskResult(success=True, data=data))

				if data:
					sources = set(d["source"] for d in data)
					logger.info(
						f"Indexed {len(data)} entries from {Path(path).name} "
						f"({', '.join(sources)})"
					)
			except Exception as e:
				results.append(TaskResult.failed(str(e)))

		return results
