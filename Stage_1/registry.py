"""
Parser registry and main entry point.

The registry maps (extension, modality) pairs to parser functions.
Each parser function looks like this: func(path: str, config: dict) -> ParseResult

A single extension can have multiple entries for different modalities.
The first modality registered for an extension becomes its default.
"""

import logging
from pathlib import Path
from Stage_1.ParseResult import ParseResult

logger = logging.getLogger("Registry")


# ===================================================================
# THE REGISTRY
#
# Key:   (extension, modality)  e.g. (".pdf", "text"), (".pdf", "image")
# Value: parser function
#
# _MODALITY_MAP stores the default modality per extension.
# It's set automatically by register() — the first modality registered
# for an extension becomes the default.
# ===================================================================

_REGISTRY: dict[tuple[str, str], callable] = {}
_MODALITY_MAP: dict[str, str] = {}


def register(extensions: str | list[str], modality: str, func: callable):
    """
    Register a parser function for one or more extensions under a modality.
    The first modality registered for an extension becomes its default.
    """
    if isinstance(extensions, str):
        extensions = [extensions]
    for ext in extensions:
        ext = ext.lower() if ext.startswith(".") else f".{ext}"
        _REGISTRY[(ext, modality)] = func
        # First registration wins as the default modality
        if ext not in _MODALITY_MAP:
            _MODALITY_MAP[ext] = modality


def get_modality(extension: str) -> str:
    """Get the default modality for an extension. Returns 'unknown' if unregistered."""
    ext = extension.lower() if extension.startswith(".") else f".{extension}"
    return _MODALITY_MAP.get(ext, "unknown")


def get_modalities_for(extension: str) -> list[str]:
    """Get all registered modalities for an extension."""
    ext = extension.lower() if extension.startswith(".") else f".{extension}"
    return [mod for (e, mod) in _REGISTRY if e == ext]


def get_supported_extensions() -> set[str]:
    """All extensions that have at least one registered parser."""
    return {ext for ext, _ in _REGISTRY}


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def parse(path: str, modality: str = None, config: dict = None, services: dict = None) -> ParseResult:
    """
    Parse a file and return standardized content.

    This is the single entry point for the entire parser system.
    Tasks call this when they need to load a file's content.

    Args:
        path:       Absolute path to the file.
        modality:   The kind of data you want back: "text", "image", "audio",
                    "video", "tabular", "container". If None, uses the default
                    modality for this file's extension.
        config:     Optional settings (max_chars, sample_rows, etc.)

    Returns:
        ParseResult with the output in the standard format for the given modality (see ParseResult).
        If no parser is registered for this (extension, modality) pair,
        returns a failed ParseResult.

    Examples:
        # Default: PDF -> text
        result = parse("report.pdf")
        print(result.output)

        # Explicit: extract images from PDF
        result = parse("report.pdf", "image")
        for img in result.output:
            img.show()

        # Chain multi-modal discovery
        result = parse("report.pdf")
        for mode in result.also_contains:
            extra = parse("report.pdf", mode)
            display(extra.output)
    """
    if config is None:
        config = {}

    path_obj = Path(path)
    extension = path_obj.suffix.lower()

    # Resolve modality
    if modality is None:
        modality = get_modality(extension)
        if modality == "unknown":
            return ParseResult(
                modality="unknown",
                metadata={"reason": f"No parser registered for {extension}"},
            )

    # Look up the parser function for this (extension, modality) pair
    parser_func = _REGISTRY.get((extension, modality))

    if parser_func is None:
        return ParseResult.failed(
            error=f"No parser for ({extension}, {modality})",
            modality=modality,
        )

    # Call the parser
    logger.debug(f"Parsing '{path_obj.name}' as {modality} (ext={extension})")
    try:
        result = parser_func(path, config, services)
        return result
    except Exception as e:
        logger.error(f"Parser failed for {path_obj.name} as {modality}: {e}")
        return ParseResult.failed(error=str(e), modality=modality)

# Initialize the registry by importing all parsers.
# Each parser file calls register() at module level, which populates
# both _REGISTRY and _MODALITY_MAP in one shot.
from Stage_1.parsers import parse_audio, parse_image, parse_tabular, parse_text, parse_video, parse_container