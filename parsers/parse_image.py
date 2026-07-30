"""Attachment parsing helpers for image inputs."""


dependencies_files = []
dependencies_pip = ['Pillow']

# Pillow decodes untrusted files in C and does its own I/O, so its actions
# cannot be turned into Requests. A process boundary is the only thing that
# actually contains it — and it is worth having, because a malformed image
# that takes down a box is a failed parse rather than a dead kernel.
isolation = "subprocess"

from PIL import Image

from guest.parsing import ParseResult, register

# Safety cap against decompression bombs. 200 MP covers the largest current
# phone sensors (e.g. Samsung 200 MP) while bounding decode memory to ~800 MB.
Image.MAX_IMAGE_PIXELS = 200_000_000


def parse_standard_image(sdk, path: str, config: dict = None) -> ParseResult:
    """Open a standard image file and return it as a PIL Image.

    The result is a live object, so it can only be used *inside* the box that
    imported this parser — put the code that consumes it in the same box and
    let text or a written file be what leaves.
    """
    try:
        img = Image.open(path)
        img.load()
        return ParseResult(
            modality="image",
            output=[img],
            metadata={"width": img.width, "height": img.height, "mode": img.mode, "format": img.format},
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="image")


register([
    ".png", ".jpg", ".jpeg", ".webp",
    ".tif", ".tiff", ".bmp", ".ico", ".gif",
], "image", parse_standard_image)


def parse_heic(sdk, path: str, config: dict = None) -> ParseResult:
    """Parse HEIC/HEIF images. Requires pillow-heif."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return parse_standard_image(sdk, path, config)
    except ImportError:
        sdk.log("pillow-heif not installed", level="debug")
        return ParseResult.failed("pillow-heif not installed", modality="image")
    except Exception as e:
        sdk.log(f"Failed to parse HEIC/HEIF {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="image")


register([".heic", ".heif"], "image", parse_heic)
