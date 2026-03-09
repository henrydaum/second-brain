"""
Embed Images task.

Loads image files with PIL, batches them, and encodes them using the
image_embedder service (e.g. CLIP). One embedding per image.

No upstream dependencies — images don't need text extraction first.
Requires the image_embedder service to be loaded.
"""

import logging
import time
from pathlib import Path

from PIL import Image

from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("EmbedImages")


class EmbedImages(BaseTask):
	name = "embed_images"
	version = 1
	modalities = ["image"]
	depends_on = []
	requires_services = ["image_embedder"]
	output_tables = ["image_embeddings"]
	output_schema = """
		CREATE TABLE IF NOT EXISTS image_embeddings (
			path TEXT PRIMARY KEY,
			embedding BLOB,
			model_name TEXT,
			embedded_at REAL
		);
	"""
	batch_size = 4
	max_workers = 4
	timeout = 300

	def run(self, paths, context):
		embedder = context.services.get("image_embedder")
		if not embedder or not embedder.loaded:
			return [TaskResult.failed("image_embedder service not loaded") for _ in paths]

		now = time.time()

		# --- 1. Load images ---
		images = []
		valid_paths = []
		failed = {}  # path -> error

		Image.MAX_IMAGE_PIXELS = None

		for path in paths:
			try:
				with Image.open(path) as img:
					img = img.convert("RGB")
					img.thumbnail((512, 512))
					images.append(img.copy())
				valid_paths.append(path)
			except Exception as e:
				logger.error(f"Failed to load image {Path(path).name}: {e}")
				failed[path] = str(e)

		# --- 2. Encode all valid images at once ---
		embeddings = None
		if images:
			try:
				embeddings = embedder.encode(images)
			except Exception as e:
				logger.error(f"Image embedding failed: {e}")
				return [TaskResult.failed(f"Encode failed: {e}") for _ in paths]

			if embeddings is None:
				return [TaskResult.failed("Embedder returned None") for _ in paths]

		# --- 3. Build results ---
		results = []
		embed_idx = 0

		for path in paths:
			if path in failed:
				results.append(TaskResult.failed(failed[path]))
				continue

			embedding_bytes = embeddings[embed_idx].tobytes()
			embed_idx += 1

			results.append(TaskResult(
				success=True,
				data=[{
					"path": path,
					"embedding": embedding_bytes,
					"model_name": embedder.model_name,
					"embedded_at": now,
				}],
			))

		logger.info(f"Embedded {len(valid_paths)} images")
		return results
