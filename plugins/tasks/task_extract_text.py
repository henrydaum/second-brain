"""
Extract Text task.

Parses text files and stores the content. Uses the Stage 1 parser system.
This is the foundation that downstream tasks (embedding, summarization)
depend on. No services required — just calls the parser directly.
"""

import logging
import time
from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("ExtractText")


class ExtractText(BaseTask):
	"""Extract text."""
	name = "extract_text"
	modalities = ["text"]
	reads = []
	writes = ["extracted_text"]
	requires_services = []  # no models needed
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

	def run(self, paths, context):
		"""Run extract text."""
		results = []
		for path in paths:
			try:
				parse_result = context.services.get("parser").parse(path, "text")

				if not parse_result.success:
					results.append(TaskResult.failed(f"Parse failed: {parse_result.error}"))
					continue

				content = parse_result.output or ""
				logger.info(f"Extracted {len(content)} chars from {Path(path).name}: {content[:25]}...")

				results.append(TaskResult(
					success=True,
					data=[{
						"path": path,
						"content": content,
						"char_count": len(content),
						"also_contains": ",".join(parse_result.also_contains),
						"extracted_at": time.time(),
					}],
					also_contains=parse_result.also_contains,
				))
			except Exception as e:
				results.append(TaskResult.failed(str(e)))
		return results