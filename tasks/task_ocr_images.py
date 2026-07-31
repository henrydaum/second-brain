"""Read text out of images with the platform's OCR engine.

Two kinds of file arrive here. A plain screenshot is itself an image. A PDF or
a slide deck is not — it reaches this task because ``extract_text`` reported
``also_contains = ["image"]``, and the pictures have to be pulled out of it
first. Both routes end at the same place: a PNG on disk that the OCR service
opens.

**The parser runs in this box, because its result cannot leave one.** A parsed
image is a live PIL object; it has no wire representation and never will. So
this task declares ``parse_modalities = ["image"]``, the kernel resolves that
against whichever parser packages are installed and loads them here before the
task runs, and ``sdk.parse.file`` calls one directly. Declaring rather than
importing the parser file is what lets the kernel see that foreign code is
being provisioned and subprocess this task accordingly.

The image then reaches the service as a *path*: scratch space is a Request the
kernel always grants, and a file is something both boxes can name. Handing over
the PIL object instead would be handing over exactly the thing that cannot
cross.
"""

dependencies_files = ['services/service_ocr.py']
dependencies_pip = []

#: The kernel loads every installed parser that produces images — raster
#: formats, PDF pages, embedded pictures in Office files — into this box.
#: Which ones exist depends on what is installed, and that is deliberately not
#: this task's business to know.
parse_modalities = ["image"]

import time

from guest.bases import BaseTask
from guest.parsing import basename

#: Long edge an image is scaled to before OCR. Engines choke on very large
#: inputs, and text that survives past this is not being read anyway.
MAX_DIMENSION = 2500


class OCRImages(BaseTask):
    """Extract text from every image in a file."""

    name = "ocr_images"
    modalities = ["image"]
    reads = []
    writes = ["ocr_text"]
    requires_services = ["ocr"]
    requests = ["parse.file", "service.call", "fs.temp", "fs.delete"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS ocr_text (
            path TEXT PRIMARY KEY,
            content TEXT,
            char_count INTEGER,
            model_name TEXT,
            extracted_at REAL
        );
    """
    batch_size = 4
    # OCR is CPU-heavy on every platform; a second worker makes both slower.
    max_workers = 1
    timeout = 300

    def run(self, sdk, paths):
        """OCR each path's images."""
        # Asked once for the batch: the adapter is named after the service, so
        # reading ``model_name`` off it would record "ocr" against every row
        # instead of the engine that actually read the text.
        described = sdk.services.call("ocr", "describe") or {}
        engine = described.get("model_name") or "unknown"

        now = time.time()
        outcomes = []

        for path in paths:
            try:
                text = self._read(sdk, path)
            except sdk.Failed as failed:
                outcomes.append({"ok": False, "error": str(failed)})
                continue

            sdk.log(f"OCR extracted {len(text)} chars from {basename(path)}"
                    if text else f"OCR found no text in {basename(path)}")

            outcomes.append({
                "ok": True,
                "data": [{
                    "path": path,
                    "content": text,
                    "char_count": len(text),
                    "model_name": engine,
                    "extracted_at": now,
                }],
            })

        return sdk.ok(per_path=outcomes)

    @staticmethod
    def _read(sdk, path) -> str:
        """Every image in one file, read and joined."""
        images = sdk.parse.file(path, "image") or []

        found = []
        for image in images:
            scratch = ""
            try:
                # Normalized the way the OCR service wants it: bounded size,
                # RGB, PNG. Doing it here rather than in the service keeps the
                # service ignorant of where its images came from.
                if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
                    image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
                if image.mode != "RGB":
                    image = image.convert("RGB")

                scratch = sdk.fs.temp(suffix=".png")
                image.save(scratch, format="PNG")

                text = sdk.services.call("ocr", "process_image",
                                         image_path=scratch)
                if text and text.strip():
                    found.append(text.strip())
            finally:
                if scratch:
                    try:
                        sdk.fs.delete(scratch)
                    except sdk.Failed:
                        sdk.log(f"could not clean up {scratch}", level="debug")

        return "\n\n".join(found).strip()
