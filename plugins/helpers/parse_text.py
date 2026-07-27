"""Plain-text and code parser — the one parser that ships in the kernel.

Reads any UTF-8 (falling back to latin-1) text or code file and returns a
standardized ``ParseResult(modality="text")``. This is the kernel's minimal
parsing floor; richer document parsers (PDF, Office, Google Drive, audio,
image, …) are installable packages that drop their own ``parse_*.py`` helper
into ``services/helpers`` and are picked up by the parser service's discovery
scan.
"""

import logging
from pathlib import Path

from plugins.services.helpers.ParseResult import ParseResult
from plugins.services.helpers import parser_registry as registry
from plugins.services.helpers.parsing_utils import clean_text, max_chars

logger = logging.getLogger("ParseText")


# Extensions whose indentation is meaningful and must be preserved.
_CODE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp", ".java", ".cs", ".php", ".rb",
    ".go", ".rs", ".swift", ".kt", ".sql", ".sh", ".bat", ".ps1",
    ".r", ".m", ".scala", ".lua", ".json", ".yaml", ".yml", ".xml",
    ".ini", ".toml", ".cfg", ".env", ".log",
}


def parse_plaintext(path: str, config: dict, services: dict = None) -> ParseResult:
    """Read any UTF-8 text file. Falls back to latin-1."""
    try:
        limit = max_chars(config)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(limit)
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                content = f.read(limit)

        is_code = Path(path).suffix.lower() in _CODE_SUFFIXES
        content = clean_text(content, preserve_indent=is_code)

        return ParseResult(
            modality="text",
            output=content,
            metadata={"char_count": len(content)},
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="text")


registry.register([
    ".txt", ".md", ".markdown", ".rst", ".tex", ".log", ".rtf",
    ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".xml",
    ".ini", ".toml", ".cfg", ".env",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".htm", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp",
    ".java", ".cs", ".php", ".rb",
    ".go", ".rs", ".swift", ".kt",
    ".sql", ".sh", ".bat", ".ps1",
    ".r", ".m", ".scala", ".lua",
], "text", parse_plaintext)
