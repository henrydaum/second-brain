import logging
from pathlib import Path
from Stage_1.ParseResult import ParseResult
import Stage_1.registry as registry

logger = logging.getLogger(__name__)

# Returns standardized PIL object

"""
Image parsers.

Returns ParseResult(modality="image", image=[PIL.Image, ...]).

For standalone image files, the list has one element.
For multi-image files (PSD with layers, GIF with frames, TIFF with pages),
the list may have multiple elements — but the default is one (the composite).

The parser validates the file is a real image and returns the PIL object(s).
Downstream tasks decide what to do: OCR, CLIP embed, thumbnail, etc.
"""


# ===================================================================
# STANDARD RASTER IMAGES
# PIL handles these natively.
# ===================================================================

def parse_standard_image(path: str, config: dict) -> ParseResult:
    """Open a standard image file and return as PIL.Image."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None  # allow large images

        img = Image.open(path)
        img.load()  # force read so file handle isn't kept open

        return ParseResult(
            modality="image",
            output=[img],
            metadata={
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="image")


registry.register([
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".tif", ".tiff", ".bmp", ".ico",
], "image", parse_standard_image)


# ===================================================================
# HEIC / HEIF (Apple format)
# ===================================================================

def parse_heic(path: str, config: dict) -> ParseResult:
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


# ===================================================================
# SVG (Vector → Raster)
# ===================================================================

def parse_svg(path: str, config: dict) -> ParseResult:
    """Rasterize SVG to a PIL Image."""
    try:
        import cairosvg
        from PIL import Image
        import io

        width = config.get("svg_render_width", 1024)
        png_data = cairosvg.svg2png(url=path, output_width=width)
        img = Image.open(io.BytesIO(png_data))
        img.load()

        return ParseResult(
            modality="image",
            output=[img],
            metadata={
                "width": img.width,
                "height": img.height,
                "format": "SVG (rasterized)",
                "original_path": path,
            },
        )
    except ImportError:
        logger.debug("cairosvg not installed")
        return ParseResult.failed("cairosvg not installed", modality="image")
    except Exception as e:
        logger.debug(f"Failed to parse SVG {path}: {e}")
        return ParseResult.failed(str(e), modality="image")


registry.register(".svg", "image", parse_svg)


# ===================================================================
# PSD (Photoshop)
# PIL opens PSD as the flattened composite.
# ===================================================================

def parse_psd(path: str, config: dict) -> ParseResult:
    """Extract the flattened composite from a PSD file."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None

        img = Image.open(path)
        img.load()

        return ParseResult(
            modality="image",
            output=[img],
            metadata={
                "width": img.width,
                "height": img.height,
                "format": "PSD (flattened)",
            },
        )
    except Exception as e:
        logger.debug(f"Failed to parse PSD {path}: {e}")
        return ParseResult.failed(str(e), modality="image")


registry.register(".psd", "image", parse_psd)