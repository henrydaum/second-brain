"""
Embed Images task.

Uses the parser system to extract images from any file type (PDF, DOCX,
standalone images, etc.), then batches and encodes them using the
image_embedder service (e.g. CLIP). One embedding per extracted image.

No upstream dependencies — images don't need text extraction first.
Requires the image_embedder service to be loaded.
"""

import logging
import time
from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("EmbedImages")


class EmbedImages(BaseTask):
	"""Embed images."""
	name = "embed_images"
	modalities = ["image"]
	reads = []
	writes = ["image_embeddings"]
	requires_services = ["image_embedder"]
	output_schema = """
		CREATE TABLE IF NOT EXISTS image_embeddings (
			path TEXT,
			image_index INTEGER,
			embedding BLOB,
			model_name TEXT,
			embedded_at REAL,
			PRIMARY KEY (path, image_index)
		);
	"""
	batch_size = 12
	max_workers = 4
	timeout = 300

	def run(self, paths, context):
		"""Run embed images."""
		embedder = context.services.get("image_embedder")
		if not embedder or not embedder.loaded:
			return [TaskResult.failed("image_embedder service not loaded") for _ in paths]

		now = time.time()

		# --- 1. Extract images via parser system ---
		# Each entry: (path, image_index, PIL.Image)
		image_entries = []
		failed = {}  # path -> error

		for path in paths:
			try:
				parse_result = context.services.get("parser").parse(path, "image")

				if not parse_result.success:
					logger.error(f"Image parse failed for {Path(path).name}: {parse_result.error}")
					failed[path] = parse_result.error
					continue

				images = parse_result.output or []
				if not images:
					logger.info(f"No images found in {Path(path).name}")
					failed[path] = "No images found"
					continue

				for idx, img in enumerate(images):
					img = img.convert("RGB")
					img.thumbnail((512, 512))
					image_entries.append((path, idx, img))

			except Exception as e:
				logger.error(f"Failed to extract images from {Path(path).name}: {e}")
				failed[path] = str(e)

		# --- 2. Encode all images at once ---
		embeddings = None
		if image_entries:
			logger.debug(f"Encoding {len(image_entries)} images across {len(paths)} file(s)...")
			try:
				pil_images = [entry[2] for entry in image_entries]
				embeddings = embedder.encode(pil_images)
			except Exception as e:
				logger.error(f"Image embedding failed: {e}")
				return [TaskResult.failed(f"Encode failed: {e}") for _ in paths]

			if embeddings is None:
				return [TaskResult.failed("Embedder returned None") for _ in paths]

		# --- 3. Build results ---
		# Group embeddings by path
		path_rows = {path: [] for path in paths}
		for i, (path, idx, _img) in enumerate(image_entries):
			path_rows[path].append({
				"path": path,
				"image_index": idx,
				"embedding": embeddings[i].tobytes(),
				"model_name": embedder.model_name,
				"embedded_at": now,
			})

		results = []
		for path in paths:
			if path in failed:
				results.append(TaskResult.failed(failed[path]))
			else:
				results.append(TaskResult(success=True, data=path_rows[path]))

		embedded_count = len(image_entries)
		file_count = len(paths) - len(failed)
		logger.info(f"Embedded {embedded_count} images from {file_count} file(s)")
		return results
