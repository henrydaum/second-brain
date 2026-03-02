"""
Parser registry and main entry point.

The registry maps (extension, modality) pairs to parser functions.
Each parser function looks like this: func(path: str, config: dict) -> ParseResult

A single extension can have multiple entries for different modalities.
The default modality for each extension is defined in _MODALITY_MAP.
"""

import logging
from pathlib import Path
from Stage_1.ParseResult import ParseResult

logger = logging.getLogger(__name__)


# ===================================================================
# THE REGISTRY
#
# Key:   (extension, modality)  e.g. (".pdf", "text"), (".pdf", "image")
# Value: parser function
# ===================================================================

_REGISTRY: dict[tuple[str, str], callable] = {}


def register(extensions: str | list[str], modality: str, func: callable):
    """Register a parser function for one or more extensions under a modality."""
    if isinstance(extensions, str):
        extensions = [extensions]
    for ext in extensions:
        ext = ext.lower() if ext.startswith(".") else f".{ext}"
        _REGISTRY[(ext, modality)] = func


def get_modalities_for(extension: str) -> list[str]:
    """Get all registered modalities for an extension."""
    ext = extension.lower() if extension.startswith(".") else f".{extension}"
    return [mod for (e, mod) in _REGISTRY if e == ext]


def get_supported_extensions() -> set[str]:
    """All extensions that have at least one registered parser."""
    return {ext for ext, _ in _REGISTRY}


# ===================================================================
# MODALITY MAP
#
# Static mapping of extension -> modality.
# Serves two purposes:
#   1. The crawler uses this to classify files WITHOUT importing parsers.
#   2. Defines the default modality for each extension.
# ===================================================================

_MODALITY_MAP: dict[str, str] = {}


def _register_modalities(extensions: list[str], modality: str):
    for ext in extensions:
        ext = ext.lower() if ext.startswith(".") else f".{ext}"
        _MODALITY_MAP[ext] = modality


# --- Text ---
_register_modalities([
    ".txt", ".md", ".markdown", ".rst", ".tex", ".log",
    ".pdf", ".docx", ".doc", ".rtf", ".pptx", ".odt", ".epub",
    ".json", ".yaml", ".yml", ".xml",
    ".ini", ".toml", ".cfg", ".env",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".htm", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp",
    ".java", ".cs", ".php", ".rb",
    ".go", ".rs", ".swift", ".kt",
    ".sql", ".sh", ".bat", ".ps1",
    ".r", ".m", ".scala", ".lua",
], "text")

# --- Image ---
_register_modalities([
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".heic", ".heif", ".tif", ".tiff",
    ".bmp", ".ico", ".svg", ".psd",
], "image")

# --- Audio ---
_register_modalities([
    ".mp3", ".wav", ".flac", ".m4a",
    ".aac", ".ogg", ".wma", ".opus",
], "audio")

# --- Tabular ---
_register_modalities([
    ".csv", ".tsv",
    ".xlsx", ".xls",
    ".parquet", ".feather",
    ".sqlite", ".db",
], "tabular")

# --- Container ---
_register_modalities([
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".eml",
], "container")

# --- Video ---
_register_modalities([
    ".mp4", ".mkv", ".avi", ".mov",
    ".webm", ".flv", ".wmv",
], "video")


def get_default_modality(extension: str) -> str:
    """Get the default modality for an extension. Returns 'unknown' if unregistered."""
    ext = extension.lower() if extension.startswith(".") else f".{extension}"
    return _MODALITY_MAP.get(ext, "unknown")


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def parse(path: str, modality: str = None, config: dict = None) -> ParseResult:
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
        modality = get_default_modality(extension)
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
    try:
        result = parser_func(path, config)
        return result
    except Exception as e:
        logger.error(f"Parser failed for {path_obj.name} as {modality}: {e}")
        return ParseResult.failed(error=str(e), modality=modality)