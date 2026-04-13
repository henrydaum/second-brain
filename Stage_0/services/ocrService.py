import asyncio
import os
import tempfile
import logging
from pathlib import Path

from Stage_0.BaseService import BaseService

# 3rd Party
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener() # Enables opening .heic/.heif
except ImportError:
    pass

logger = logging.getLogger("OCRClass")

class WindowsOCR(BaseService):
    def __init__(self):
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

        # 1. PRE-PROCESS: resize and convert to RGB PNG.
        # This prevents the OCR engine from choking on huge/weird files.
        temp_path = self._create_optimized_temp_file(image_path)
        if not temp_path:
            return ""

        try:
            # 2. RUN OCR via Windows async API (blocking wrapper).
            # This can hang if the Windows OCR engine is unresponsive.
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

    def _create_optimized_temp_file(self, original_path):
        """
        Your PIL helper. Essential for stability.
        """
        try:
            with Image.open(original_path) as img:
                # Resize to safe max dimension
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


def build_services(config: dict) -> dict:
    import platform
    if platform.system() != "Windows":
        logger.info("OCR service skipped (Windows-only).")
        return {}
    return {"ocr": WindowsOCR()}