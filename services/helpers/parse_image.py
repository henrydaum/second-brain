"""Attachment parsing helpers for image inputs."""


dependencies_files = []
dependencies_pip = ['Pillow']

import logging
from PIL import Image
from plugins.services.helpers.ParseResult import ParseResult
from plugins.services.helpers import parser_registry as registry

logger = logging.getLogger("ParseImage")

# Safety cap against decompression bombs. 200 MP covers the largest current
# phone sensors (e.g. Samsung 200 MP) while bounding decode memory to ~800 MB.
Image.MAX_IMAGE_PIXELS = 200_000_000


def parse_standard_image(path: str, config: dict, services: dict = None) -> ParseResult:
    """Open a standard image file and return as PIL.Image."""
    try:
        img = Image.open(path)
        img.load()
        return ParseResult(
            modality="image",
            output=[img],
            metadata={"width": img.width, "height": img.height, "mode": img.mode, "format": img.format},
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="image")


registry.register([
    ".png", ".jpg", ".jpeg", ".webp",
    ".tif", ".tiff", ".bmp", ".ico", ".gif",
], "image", parse_standard_image)


def parse_heic(path: str, config: dict, services: dict = None) -> ParseResult:
    """Parse HEIC/HEIF images. Requires pillow-heif."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return parse_standard_image(path, config)
    except ImportError:
        logger.debug("pillow-heif not installed")
        return ParseResult.failed("pillow-heif not installed", modality="image")
    except Exception as e:
        logger.debug(f"Failed to parse HEIC/HEIF {path}: {e}")
        return ParseResult.failed(str(e), modality="image")


registry.register([".heic", ".heif"], "image", parse_heic)
