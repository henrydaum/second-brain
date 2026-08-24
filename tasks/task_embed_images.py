"""Embed every image in a file, so pictures can be found by description.

Same shape as ``task_ocr_images`` and for the same reason: the parser runs in
this box because a parsed image is a live PIL object, and the embedder receives
a *path* because that is what two boxes can both name. Declared with
``parse_modalities`` so the kernel provisions the parsers and knows to contain
this task.

Images are pooled across the whole batch before encoding. One ``encode`` call
over thirty images is much faster than thirty calls over one, because the GPU
batches, and the box serializes its calls anyway — so per-file encoding would
pay the round-trip cost thirty times for no parallelism.
"""

dependencies_files = ['services/service_embed.py']
dependencies_pip = []

parse_modalities = ["image"]

import time

from guest.bases import BaseTask
from guest.parsing import basename

#: CLIP sees 224px; anything past this is scaled away before it reaches the
#: model, so writing it to scratch first is wasted bytes.
MAX_DIMENSION = 512


class EmbedImages(BaseTask):
    """One vector per image found in a file."""

    name = "embed_images"
    description = (
        "Embed every image in a file into a vector, so pictures can be "
        "found by describing them.")
    modalities = ["image"]
    reads = []
    writes = ["image_embeddings"]
    requires_services = ["image_embedder"]
    requests = ["parse.file", "service.call", "fs.temp", "fs.delete"]
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

    def run(self, sdk, paths):
        """Extract every image in the batch, then encode them in one go."""
        described = sdk.services.call("image_embedder", "describe") or {}
        model_name = described.get("model_name") or "unknown"
        now = time.time()

        # ── 1. Extract, per file, so one bad file fails alone ──────────
        staged = {}          # path -> [scratch png, ...]
        failures = {}        # path -> why
        for path in paths:
            try:
                staged[path] = self._stage(sdk, path)
            except sdk.Failed as failed:
                failures[path] = str(failed)

        pooled = [scratch for path in paths
                  for scratch in staged.get(path, ())]

        # ── 2. Encode the pool in one call ─────────────────────────────
        vectors = []
        if pooled:
            try:
                sdk.log(f"encoding {len(pooled)} image(s) from "
                        f"{len(staged)} file(s)")
                vectors = sdk.services.call("image_embedder", "encode",
                                            inputs=pooled) or []
            except sdk.Failed as failed:
                # The encode is one call over the whole batch, so its failure
                # genuinely is the batch's — every file goes down together and
                # each is told the same true reason.
                return sdk.ok(per_path=[{"ok": False,
                                         "error": f"encode failed: {failed}"}
                                        for _ in paths])
            finally:
                # Runs on the way out of the return above too, so the scratch
                # files are cleaned up exactly once on both paths.
                self._discard(sdk, pooled)

        if pooled and len(vectors) != len(pooled):
            return sdk.ok(per_path=[
                {"ok": False,
                 "error": f"embedder returned {len(vectors)} vectors for "
                          f"{len(pooled)} images"} for _ in paths])

        # ── 3. Hand each file back its own slice ───────────────────────
        outcomes = []
        cursor = 0
        for path in paths:
            if path in failures:
                outcomes.append({"ok": False, "error": failures[path]})
                continue

            mine = staged.get(path, [])
            rows = [{
                "path": path,
                "image_index": index,
                "embedding": vectors[cursor + index],
                "model_name": model_name,
                "embedded_at": now,
            } for index in range(len(mine))]
            cursor += len(mine)

            sdk.log(f"embedded {len(rows)} image(s) from {basename(path)}")
            outcomes.append({"ok": True, "data": rows})

        return sdk.ok(per_path=outcomes)

    @staticmethod
    def _stage(sdk, path) -> list:
        """Write one file's images to scratch, and return their paths.

        Scaled down first. The embedder resizes to the model's input anyway,
        so a full-resolution PNG on the way there is bytes written and read
        for nothing.
        """
        written = []
        for image in sdk.parse.file(path, "image") or []:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
            scratch = sdk.fs.temp(suffix=".png")
            image.save(scratch, format="PNG")
            written.append(scratch)
        return written

    @staticmethod
    def _discard(sdk, scratches) -> None:
        """Clean up staged images. Failing to is not worth failing a batch."""
        for scratch in scratches:
            try:
                sdk.fs.delete(scratch)
            except sdk.Failed:
                sdk.log(f"could not clean up {scratch}", level="debug")
