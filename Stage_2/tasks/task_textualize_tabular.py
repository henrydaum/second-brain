"""
Textualize Tabular task.

Converts CSV/XLSX files into Markdown table representations so the LLM
can reason about tabular data. Caps at 50 rows per sheet to keep output
manageable. Stored in the tabular_text table for the metadata scraper.
"""

import logging
import time
from pathlib import Path

from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("TextualizeTabular")

MAX_ROWS = 50


class TextualizeTabular(BaseTask):
	name = "textualize_tabular"
	modalities = ["tabular"]
	reads = []
	writes = ["tabular_text"]
	requires_services = []
	output_schema = """
		CREATE TABLE IF NOT EXISTS tabular_text (
			path TEXT PRIMARY KEY,
			content TEXT,
			char_count INTEGER,
			textualized_at REAL
		);
	"""
	batch_size = 4
	timeout = 120

	def run(self, paths, context):
		results = []
		for path in paths:
			try:
				parse_result = context.services.get("parser").parse(path, "tabular")

				if not parse_result.success:
					results.append(TaskResult.failed(f"Parse failed: {parse_result.error}"))
					continue

				sheets = parse_result.output or {}
				sections = []

				for sheet_name, df in sheets.items():
					truncated = df.head(MAX_ROWS)
					try:
						md = truncated.to_markdown(index=False)
					except ImportError:
						md = truncated.to_string(index=False)

					header = f"## {sheet_name}" if len(sheets) > 1 else ""
					if len(df) > MAX_ROWS:
						footer = f"\n... ({len(df) - MAX_ROWS} more rows)"
					else:
						footer = ""

					section = "\n".join(part for part in [header, md, footer] if part)
					sections.append(section)

				content = "\n\n".join(sections)
				logger.info(f"Textualized {len(content)} chars from {Path(path).name}")

				results.append(TaskResult(
					success=True,
					data=[{
						"path": path,
						"content": content,
						"char_count": len(content),
						"textualized_at": time.time(),
					}],
				))
			except Exception as e:
				logger.error(f"Textualize failed for {Path(path).name}: {e}")
				results.append(TaskResult.failed(str(e)))
		return results
