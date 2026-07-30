"""Optical character recognition, using whatever engine this machine has.

Three engines, one service. The native version was three ``BaseService``
subclasses and a ``build_services`` that picked one by platform, which made
sense when a file could hold several classes for the kernel to choose between.
It cannot mean that any more: a file declaring several services now registers
*all* of them and gives them one shared box (see ``sandbox/bridge.py``), which
is the opposite of what this file wants. Only one engine is ever usable here,
so choosing is something ``start()`` does, once, from
``sdk.paths.get("platform")``.

**No engine ships with this package.** They are platform-specific, large, or
both — Windows OCR is a set of WinRT projections, macOS Vision is resident in
the OS behind pyobjc, Linux uses EasyOCR and torch. Declaring all three in
``dependencies_pip`` would make ``pip install`` fail on every platform, since
each list is uninstallable on the other two. So the package brings only the
image handling, and a missing engine is reported with the exact command to
install it.
"""

dependencies_files = []
dependencies_pip = ['pillow', 'pillow-heif']

import asyncio

from guest.bases import BaseService

#: The longest edge an image is scaled to before OCR. Engines choke on very
#: large inputs, and text that survives past this is not being read anyway.
MAX_DIMENSION = 2500

#: ``sys.platform`` prefix -> (engine label, pip install line).
ENGINES = {
    "win32": (
        "winrt_windows_ocr",
        "winrt-Windows.Media.Ocr winrt-Windows.Graphics.Imaging "
        "winrt-Windows.Storage",
    ),
    "darwin": (
        "apple_vision_ocr",
        "pyobjc-framework-Vision pyobjc-framework-Quartz "
        "pyobjc-framework-Cocoa",
    ),
    "linux": ("easyocr", "easyocr torch"),
}


class OCRService(BaseService):
    """Read text out of images, with this platform's OCR engine."""

    name = "ocr"
    description = "Extract text from images using the platform's OCR engine."
    shared = True
    timeout = 300
    requests = ["paths.get", "fs.temp", "fs.delete", "fs.list"]
    exports = ["process_image", "describe"]

    def __init__(self):
        """Nothing is chosen until start()."""
        self.engine = ""
        self.platform = ""
        self.reader = None          # EasyOCR only; the others are stateless

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Pick an engine for this platform and check it actually imports.

        Verifying at load rather than at first use is deliberate: an OCR
        service that loads cleanly and then fails every image is a service the
        user has no reason to look at. Failing here means ``/services`` shows
        it as not loaded, with the reason.
        """
        try:
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(_missing("Pillow", "pillow", exc))
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()   # enables .heic / .heif
        except ImportError:
            pass    # HEIC support is a bonus, not a requirement

        self.platform = str(sdk.paths.get("platform") or "")
        matched = next((key for key in ENGINES
                        if self.platform.startswith(key)), "")
        if not matched:
            sdk.log(f"no OCR engine for platform {self.platform!r}",
                    level="warning")
            return False

        self.engine, install = ENGINES[matched]
        starter = {"win32": self._start_windows, "darwin": self._start_mac,
                   "linux": self._start_linux}[matched]
        starter(sdk, install)
        sdk.log(f"OCR ready: {self.engine}")
        return True

    def _start_windows(self, sdk, install):
        """Check the WinRT projections import."""
        # torch loads DLLs greedily and collides with winrt's if it lands
        # second, so import it first *when it happens to be installed*.
        # Windows OCR does not need it; its absence is fine.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        try:
            from winrt.windows.media.ocr import OcrEngine  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(_missing("Windows OCR via winrt", install, exc))

    def _start_mac(self, sdk, install):
        """Check the Vision frameworks import; the model is OS-resident."""
        try:
            import Quartz  # noqa: F401
            import Vision  # noqa: F401
            from Foundation import NSURL  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(_missing("macOS Vision", install, exc))

    def _start_linux(self, sdk, install):
        """Build the EasyOCR reader, which downloads weights on first use."""
        try:
            import easyocr
            import torch
        except ImportError as exc:
            raise RuntimeError(_missing("EasyOCR", install, exc))
        gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        sdk.log(f"EasyOCR initialized (gpu={gpu})")

    def stop(self, sdk):
        """Release the reader. The stateless engines have nothing to drop."""
        self.reader = None
        return None

    # ── exports ─────────────────────────────────────────────────────

    def describe(self, sdk):
        """Which engine is loaded.

        The bridge names an adapter after the service, so a caller reading
        ``ocr.model_name`` would get "ocr". Every OCR row records the engine
        that produced it, so it has to be askable.
        """
        return {
            "model_name": self.engine,
            "engine": self.engine,
            "platform": self.platform,
            "loaded": bool(self.engine),
        }

    def process_image(self, sdk, image_path):
        """Read text out of one image file. Returns "" when there is none."""
        if not self.engine:
            return ""
        if not _exists(sdk, image_path):
            sdk.log(f"image not found: {image_path}", level="warning")
            return ""

        name = sdk.path.name(image_path)
        prepared = self._prepare(sdk, image_path)
        if not prepared:
            return ""

        try:
            if self.platform.startswith("win32"):
                return asyncio.run(self._read_windows(sdk, prepared)) or ""
            if self.platform.startswith("darwin"):
                return self._read_mac(sdk, prepared) or ""
            return self._read_linux(sdk, prepared) or ""
        except Exception as exc:
            sdk.log(f"OCR failed for {name}: {exc}", level="warning")
            return ""
        finally:
            try:
                sdk.fs.delete(prepared)
            except Exception:
                sdk.log(f"could not clean up {prepared}", level="debug")

    # ── preparation ─────────────────────────────────────────────────

    def _prepare(self, sdk, original):
        """Normalize an image into a scratch PNG the engines can all read.

        Scratch space is a Request the kernel always grants, which is what
        makes this cost no dialog — unlike ``tempfile``, which reaches the
        filesystem directly and the validator refuses.
        """
        from PIL import Image

        try:
            scratch = sdk.fs.temp(suffix=".png")
            with Image.open(original) as img:
                if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
                    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION),
                                  Image.Resampling.LANCZOS)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(scratch, format="PNG")
            return scratch
        except Exception as exc:
            sdk.log(f"image preprocess failed for "
                    f"{sdk.path.name(original)}: {exc}", level="error")
            return ""

    # ── the three engines ───────────────────────────────────────────

    async def _read_windows(self, sdk, path):
        """Windows OCR. The engine is built per call, on this thread."""
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage import StorageFile

        handle = await StorageFile.get_file_from_path_async(path)
        stream = await handle.open_async(0)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()

        # Created per call rather than held: a shared engine across threads
        # crashes, and building one is cheap next to decoding the image.
        engine = OcrEngine.try_create_from_user_profile_languages()
        if not engine:
            return ""
        result = await engine.recognize_async(bitmap)
        return "\n".join(line.text for line in result.lines)

    def _read_mac(self, sdk, path):
        """Apple Vision, via VNRecognizeTextRequest."""
        import Quartz
        import Vision
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(path)
        source = Quartz.CGImageSourceCreateWithURL(url, None)
        if not source:
            return ""
        image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
        if not image:
            return ""

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        handler = (Vision.VNImageRequestHandler.alloc()
                   .initWithCGImage_options_(image, None))
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            sdk.log(f"Vision OCR error: {error}", level="error")
            return ""

        lines = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if candidates and candidates.count() > 0:
                lines.append(str(candidates.objectAtIndex_(0).string()))
        return "\n".join(lines)

    def _read_linux(self, sdk, path):
        """EasyOCR."""
        if self.reader is None:
            return ""
        found = self.reader.readtext(path, detail=0, paragraph=True)
        return "\n".join(line.strip() for line in found
                         if isinstance(line, str) and line.strip())


def _exists(sdk, path) -> bool:
    """Whether a path is there.

    ``sdk.fs.list`` *fails* on a missing path rather than answering with an
    empty list, and the SDK turns a failed Request into a raise — so
    ``if sdk.fs.list(p)`` does not test existence, it throws.
    """
    try:
        return bool(sdk.fs.list(path))
    except sdk.Failed:
        return False


def _missing(engine: str, install: str, err: Exception) -> str:
    """What to say when the platform's OCR engine is not installed.

    OCR ships without one — see the module docstring — so this turns a bare
    ImportError into the command that fixes it.
    """
    return (
        f"OCR is installed but its engine is not available on this machine "
        f"({engine}: {err}).\n"
        f"Install it with:\n"
        f"    pip install {install}\n"
        f"or ask the agent to run that for you. Then reload the OCR service "
        f"via /services (or restart)."
    )
