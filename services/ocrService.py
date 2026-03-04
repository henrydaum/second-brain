import asyncio
import os
import tempfile
import logging
from pathlib import Path

# 3rd Party
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener() # Enables opening .heic/.heif
except ImportError:
    pass

logger = logging.getLogger("OCRClass")

class WindowsOCR:
    def __init__(self):
        self.enabled = False
        self.model_name = "winrt_windows_ocr"

    @property
    def loaded(self):
        return self.enabled

    def load(self):
        """Just imports stuff and checks if library is present and enables the flag."""
        logger.info("Enabling Windows OCR")
        # Must import this before other .dlls
        import torch
        
        self.enabled = True
        logger.info("Windows OCR loaded.")
        return True

    def unload(self):
        self.enabled = False
        logger.info("Windows OCR unloaded.")

    def process_image(self, image_path):
        """
        The Orchestrator calls this. We use YOUR proven logic here.
        """
        
        if not self.enabled: return ""
        if not os.path.exists(image_path): return ""

        # 1. PRE-PROCESS (Your PIL optimization)
        # This is crucial. It prevents the engine from choking on huge/weird files.
        temp_path = self._create_optimized_temp_file(image_path)
        if not temp_path:
            return ""

        try:
            # 2. RUN OCR (Your Async Wrapper)
            # We run this BLOCKING, so the Orchestrator waits for the result.
            text = asyncio.run(self._run_windows_ocr_task(temp_path))
            return text if text else ""
        except Exception as e:
            logger.info(f"[Error] OCR Failed for {Path(image_path).name}: {e}")
            return ""
        finally:
            # 3. CLEANUP
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except: pass

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
            logger.error(f"[OCR Async Error] {e}")
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
            logger.error(f"[OCR Pre-process Error] {Path(original_path).name}: {e}")
            return None