"""
Extract Container task.

Extracts archive files (ZIP, TAR, 7Z, RAR, EML) and registers
the child files back into the system. Uses the Stage 1 container
parsers which handle extraction, security limits, and deduplication.

The extracted child paths are returned via TaskResult.discovered_paths,
which the orchestrator picks up and feeds back through on_paths_discovered.
This means child files get the full treatment: DB registration, task
queuing, and even recursive container extraction if there's an archive
inside an archive.

No services required — just calls the parser directly.
"""

import logging
import time
from pathlib import Path

from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("ExtractContainer")


class ExtractContainer(BaseTask):
	name = "extract_container"
	version = 1
	modalities = ["container"]
	depends_on = []
	requires_services = []  # no models needed
	output_tables = ["extracted_containers"]
	output_schema = """
		CREATE TABLE IF NOT EXISTS extracted_containers (
			path TEXT PRIMARY KEY,
			archive_format TEXT,
			file_count INTEGER,
			extract_dir TEXT,
			extracted_at REAL
		);
	"""
	batch_size = 2
	max_workers = 2  # extraction is I/O-heavy, limit concurrency
	timeout = 300

	def run(self, paths, context):
		results = []
		for path in paths:
			try:
				parse_result = context.parse(path, "container")

				if not parse_result.success:
					logger.warning(f"Container parse failed for {Path(path).name}: {parse_result.error}")
					results.append(TaskResult.failed(f"Parse failed: {parse_result.error}"))
					continue

				child_paths = parse_result.output or []
				metadata = parse_result.metadata

				logger.info(
					f"Extracted {len(child_paths)} files from {Path(path).name} "
					f"({metadata.get('archive_format', 'unknown')} archive)"
				)

				results.append(TaskResult(
					success=True,
					data=[{
						"path": path,
						"archive_format": metadata.get("archive_format", ""),
						"file_count": len(child_paths),
						"extract_dir": metadata.get("extract_dir", ""),
						"extracted_at": time.time(),
					}],
					discovered_paths=child_paths,
				))
			except Exception as e:
				logger.error(f"Container extraction failed for {Path(path).name}: {e}")
				results.append(TaskResult.failed(str(e)))
		return results