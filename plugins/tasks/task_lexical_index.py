"""
Lexical Index task.

Reads text chunks, OCR text, and/or tabular text and writes them to the
lexical_content table, which feeds an FTS5 full-text search index via
SQLite triggers. Enables BM25-ranked keyword search across all indexed files.

Indexes at chunk level so BM25 results align with embedding results
for hybrid search fusion. OCR and tabular text (typically short) get
chunk_index=0.

This is a downstream task — no modalities needed. It runs whenever
any upstream (chunk_text, ocr_images, or textualize_tabular) completes
for a path.

Depends on text_chunks (produced by chunk_text) OR ocr_text (produced
by ocr_images) OR tabular_text (produced by textualize_tabular).
require_all_inputs = False means any one suffices.
"""

import logging
import time
from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("IndexLexical")


class IndexLexical(BaseTask):
	name = "index_lexical"
	modalities = []  # downstream task — triggered by upstream completion
	reads = ["text_chunks", "ocr_text", "tabular_text"]
	writes = ["lexical_content"]
	require_all_inputs = False  # run when either input exists
	requires_services = []
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

				# Index tabular text if available
				tabular = context.db.get_task_output("tabular_text", path)
				if tabular:
					content = tabular[0]["content"] or ""
					if content.strip():
						data.append({
							"path": path,
							"source": "tabular",
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
