"""PDF parser. Delegates to the existing ``parser`` service if present so
we don't duplicate text-extraction logic; falls back to None otherwise."""

from attachments.registry import register

_DEFAULT_MAX = 8000


def parse_pdf(path: str, services: dict, config: dict) -> str | None:
    """Parse PDF."""
    parser = (services or {}).get("parser")
    if parser is None:
        return None
    max_chars = int(config.get("max_chars") or _DEFAULT_MAX)
    try:
        result = parser.parse(path, "text", config={"max_chars": max_chars})
    except Exception:
        return None
    output = getattr(result, "output", None)
    if isinstance(output, str) and output.strip():
        return output
    return None


register([".pdf"], "text", parse_pdf)
