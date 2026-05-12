"""Plain-text-ish file parser. Reads the file bytes as utf-8 and returns
the leading ``max_chars`` characters (default 8000)."""

from pathlib import Path

from attachments.registry import register

_DEFAULT_MAX = 8000

_EXTENSIONS = [
    ".txt", ".md", ".log", ".rtf",
    ".csv", ".tsv",
    ".json", ".yml", ".yaml", ".xml", ".toml", ".ini",
    ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".ps1", ".sql",
]


def parse_text(path: str, services: dict, config: dict) -> str | None:
    """Parse text."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    max_chars = int(config.get("max_chars") or _DEFAULT_MAX)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[...truncated, {len(text) - max_chars} more chars]"
    return text


register(_EXTENSIONS, "text", parse_text)
