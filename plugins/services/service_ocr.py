"""Service plugin for OCR."""

import asyncio
import os
import tempfile
import logging
from pathlib import Path

from plugins.BaseService import BaseService

# 3rd Party
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener() # Enables opening .heic/.heif
except ImportError:
    pass

logger = logging.getLogger("OCRClass")


def _create_optimized_temp_file(original_path):
    """Resize/normalize the image to a temp PNG so the OCR engine doesn't
    choke on huge or unusual files. Shared by all OCR backends."""
    try:
        with Image.open(original_path) as img:
            max_dim = 2500
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            # delete=False required for Windows
            temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(temp.name, format="PNG")
            temp.close()
            return temp.name
    except Exception as e:
        logger.error(f"Image preprocess failed for {Path(original_path).name}: {e}")
        return None


class WindowsOCR(BaseService):
    """Windows OCR."""
    def __init__(self):
        """Initialize the windows OCR."""
        super().__init__()
        self.model_name = "winrt_windows_ocr"
        self.shared = True

    def _load(self):
        """Just imports stuff and checks if library is present and enables the flag."""
        # Torch must be imported before winrt — its greedy DLL loading
        # conflicts with winrt's DLLs if loaded second, causing crashes.
        import torch  # noqa: F401

        self.loaded = True
        return True

    def unload(self):
        """Handle unload."""
        self.loaded = False
        logger.info("Windows OCR unloaded.")

    def process_image(self, image_path):
        """
        Run OCR on a single image file. Returns extracted text or empty string.
        Pre-processes the image first (resize, convert to RGB) for stability.
        """
        import time as _time

        if not self.loaded: return ""
        if not os.path.exists(image_path): return ""

        temp_path = _create_optimized_temp_file(image_path)
        if not temp_path:
            return ""

        try:
            logger.debug(f"OCR starting: {Path(image_path).name}")
            t0 = _time.time()
            text = asyncio.run(self._run_windows_ocr_task(temp_path))
            logger.debug(f"OCR completed: {Path(image_path).name} in {_time.time() - t0:.2f}s")
            return text if text else ""
        except Exception as e:
            logger.warning(f"OCR failed for {Path(image_path).name}: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.debug(f"Temp cleanup failed: {e}")

    async def _run_windows_ocr_task(self, image_path):
        """
        Your exact async logic. Creates engine ON THE FLY to prevent crashes.
        """
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.storage import StorageFile

        try:
            abs_path = os.path.abspath(image_path)

            # A. File Access
            f = await StorageFile.get_file_from_path_async(abs_path)
            stream = await f.open_async(0)

            # B. Decode
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()

            # C. Create Engine (Thread-Local Safety!)
            engine = OcrEngine.try_create_from_user_profile_languages()
            if not engine:
                return None

            # D. Recognize
            result = await engine.recognize_async(bitmap)
            lines = [line.text for line in result.lines]
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Async OCR failed: {e}")
            return None


class MacOCR(BaseService):
    """OCR backed by Apple's Vision framework (VNRecognizeTextRequest).

    The model ships with macOS — no separate download or system install.
    Requires `pyobjc-framework-Vision` and `pyobjc-framework-Quartz` (pip).
    """

    def __init__(self):
        """Initialize the mac OCR."""
        super().__init__()
        self.model_name = "apple_vision_ocr"
        self.shared = True

    def _load(self):
        # Verify the frameworks import; the OCR model is OS-resident.
        """Internal helper to load mac OCR."""
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
        self.loaded = True
        return True

    def unload(self):
        """Handle unload."""
        self.loaded = False
        logger.info("Mac OCR unloaded.")

    def process_image(self, image_path):
        """Handle process image."""
        import time as _time

        if not self.loaded: return ""
        if not os.path.exists(image_path): return ""

        temp_path = _create_optimized_temp_file(image_path)
        if not temp_path:
            return ""

        try:
            logger.debug(f"OCR starting: {Path(image_path).name}")
            t0 = _time.time()
            text = self._run_vision_ocr(temp_path)
            logger.debug(f"OCR completed: {Path(image_path).name} in {_time.time() - t0:.2f}s")
            return text or ""
        except Exception as e:
            logger.warning(f"OCR failed for {Path(image_path).name}: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.debug(f"Temp cleanup failed: {e}")

    def _run_vision_ocr(self, image_path):
        """Internal helper to run vision OCR."""
        import Vision
        import Quartz
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(os.path.abspath(image_path))
        src = Quartz.CGImageSourceCreateWithURL(url, None)
        if not src:
            return ""
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if not cg_image:
            return ""

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        success, error = handler.performRequests_error_([request], None)
        if not success:
            logger.error(f"Vision OCR error: {error}")
            return ""

        observations = request.results() or []
        lines = []
        for obs in observations:
            candidate = obs.topCandidates_(1)
            if candidate and candidate.count() > 0:
                lines.append(str(candidate.objectAtIndex_(0).string()))
        return "\n".join(lines)


def build_services(config: dict) -> dict:
    """Build services."""
    import platform
    system = platform.system()
    if system == "Windows":
        return {"ocr": WindowsOCR()}
    if system == "Darwin":
        return {"ocr": MacOCR()}
    logger.info(f"OCR service skipped (no backend for platform: {system}).")
    return {}
