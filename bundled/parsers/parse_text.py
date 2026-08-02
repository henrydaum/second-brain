"""Plain-text and code parser — the one parser that ships in the kernel.

Reads any UTF-8 (falling back to latin-1) text or code file and returns a
standardized ``ParseResult(modality="text")``. This is the kernel's minimal
parsing floor; richer document parsers (PDF, Office, Google Drive, audio,
image, …) are installable packages that drop their own ``parse_*.py`` file
into ``parsers/`` at a plugin tree's root, where ``parsing.discover()`` finds
them.

A parser is a *library*, not a plugin: no base class, no entry point. Code
that wants a modality whose result cannot cross a process boundary imports the
parser into its own box and calls it there.
"""

from guest.parsing import ParseResult, clean_text, max_chars, register


# Extensions whose indentation is meaningful and must be preserved.
_CODE_SUFFIXES = (
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp", ".java", ".cs", ".php", ".rb",
    ".go", ".rs", ".swift", ".kt", ".sql", ".sh", ".bat", ".ps1",
    ".r", ".m", ".scala", ".lua", ".json", ".yaml", ".yml", ".xml",
    ".ini", ".toml", ".cfg", ".env", ".log",
)


def parse_plaintext(sdk, path: str, config: dict = None) -> ParseResult:
    """Read any UTF-8 text file. Falls back to latin-1.

    The read goes through ``sdk.fs.read`` rather than ``open``, which is what
    makes this file importable into a box: inside one the read is a mediated
    Request, and here it is the kernel reading its own disk. The parser does
    not know or care which — that is the point of the shared signature.
    """
    try:
        content = sdk.fs.read(path)[:max_chars(config)]

        # String matching rather than pathlib: a parser must load in a
        # subprocess box, where importing anything that reaches the
        # environment is refused — and a suffix is just the end of a string.
        is_code = path.lower().endswith(_CODE_SUFFIXES)
        content = clean_text(content, preserve_indent=is_code)

        return ParseResult(
            modality="text",
            output=content,
            metadata={"char_count": len(content)},
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="text")


register([
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
