"""
Extract Text task.

A simple first task that parses text files and stores the content.
Uses the Stage 1 parser system to read files.

This is a real task, not a dummy — it's the foundation that
downstream tasks (embedding, summarization) will depend on.
"""

import logging
import time
from pathlib import Path

from Stage_1.registry import parse
from Stage_2.BaseTask import BaseTask, TaskContext, TaskResult

logger = logging.getLogger(__name__)


class ExtractText(BaseTask):
	name = "extract_text"
	version = 1
	modalities = ["text"]
	depends_on = []
	output_tables = ["extracted_text"]
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

	def run(self, paths, context):
		results = []
		for path in paths:
			try:
				parse_result = parse(path, "text")

				if not parse_result.success:
					results.append(TaskResult.failed(f"Parse failed: {parse_result.error}"))
					continue
				
				print(f"Found content: {parse_result.output[:100]}...")  # Debug log

				content = parse_result.output or ""

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