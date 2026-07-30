"""The parser contract — what a ``parse_*.py`` helper returns.

This lives in the guest because a parser is *guest code*. It runs inside a
box, alongside whatever consumes its output, and the shape it hands back is
part of what a plugin author writes against — the same reason
:mod:`guest.bases` and :mod:`guest.hooks` are here rather than in the kernel.

The practical consequence is the whole reason it moved: a parser importing a
kernel module loads in-process and fails in a subprocess, which is the case
the heavy parsers most need. Importing the contract from the guest resolves
identically in both. (CLAUDE.md, "The contract lives in the guest".)

A parser is::

    def parse_x(sdk, path, config=None) -> ParseResult

One signature, two callers. Inside a box ``sdk`` is the real SDK and every
effect is a Request; when the kernel calls it, ``sdk`` is an in-process
stand-in that reads directly. The parser cannot tell, and does not need to.

Stdlib only, like everything else in here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Payloads that are the *result* of parsing rather than an intermediate step,
# and therefore the only ones that can leave the box that produced them.
#
# Every other modality resolves to a live object from a foreign library — a
# PIL image, a numpy array, an open ``av.Container`` — on its way to text or
# to a file. Those belong wherever the code that consumes them lives: import
# the parser into that box and the object never has to travel.
CROSSABLE = frozenset({"text", "container"})

# ~125k tokens. Generous default ceiling for text extraction.
DEFAULT_MAX_CHARS = 500_000


@dataclass
class ParseResult:
    """One file, parsed into a standard shape.

    ``output`` carries the payload, and its type follows the modality::

        text      -> str (UTF-8)
        container -> list[str]                    # extracted child paths
        image     -> parser-defined image objects
        audio     -> tuple(np.ndarray, int)       # (samples, sample_rate)
        video     -> av.Container
        tabular   -> dict[sheet_name or "default", pd.DataFrame]
    """

    modality: str = "unknown"
    success: bool = True
    error: str = ""

    output: Any = None

    # Lightweight, always populated.
    metadata: dict = field(default_factory=dict)

    # Multi-modal discovery — what else is in this file? e.g. ["image",
    # "tabular"] for a PDF with charts and photos. This is how one parse tells
    # the pipeline there is another route out of the same file.
    also_contains: list = field(default_factory=list)

    @staticmethod
    def failed(error: str, modality: str = "unknown") -> "ParseResult":
        """Convenience constructor for parse failures."""
        return ParseResult(success=False, error=error, modality=modality)

    @property
    def crossable(self) -> bool:
        """Whether this payload can leave the process that produced it."""
        return self.modality in CROSSABLE


# ──────────────────────────────────────────────────────────────────────
# Shared text helpers. Pure: no Request, no cost, and every parser — heavy
# or light — leans on them, so they ship with the contract rather than
# sitting somewhere a subprocessed parser cannot reach.
# ──────────────────────────────────────────────────────────────────────

def basename(path: str) -> str:
    """The file name part of a path, for messages.

    A pure string operation. ``pathlib`` is refused in sandboxed code because
    ``Path`` is one attribute away from touching the disk, but *inspecting* a
    path is not an effect — so the safe part lives here rather than being
    reinvented, slightly wrong, in every parser.
    """
    return (path or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def suffix(path: str) -> str:
    """The lower-cased extension of a path, dot included, or "" if none."""
    name = basename(path)
    return f".{name.rsplit('.', 1)[-1].lower()}" if "." in name[1:] else ""


def max_chars(config: dict | None) -> int:
    """Return the configured character limit for text parsing."""
    return (config or {}).get("max_chars", DEFAULT_MAX_CHARS)


def clean_text(text: str, preserve_indent: bool = False) -> str:
    """Normalize whitespace and remove junk.

    If preserve_indent is True, only collapse horizontal whitespace within
    lines (not leading whitespace), keeping indentation intact.
    """
    if not text:
        return ""
    if preserve_indent:
        # Collapse runs of spaces/tabs mid-line only, keep leading whitespace
        text = re.sub(r"(?<=\S)[ \t]+", " ", text)
    else:
        text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# Declaring what a parser handles.
#
# ``register`` is a *collector*, not a registry. A parser calls it at import
# time and the kernel drains what accumulated; a box importing the same file
# runs the same line and nothing happens, because a box has no registry to
# write to and does not need one — it imported the parse function by name.
#
# That is what keeps one file loadable in both worlds. The alternative was a
# kernel import at module scope, which is exactly the thing a child process
# cannot resolve.
# ──────────────────────────────────────────────────────────────────────

_DECLARED: list = []


def register(extensions, modality: str, func) -> None:
    """Declare that ``func`` parses these extensions as ``modality``.

    The first modality declared for an extension becomes its default, so
    order matters: a PDF that declares text before image is text by default.
    """
    _DECLARED.append((extensions, modality, func))


def drain_registrations() -> list:
    """Take everything declared since the last drain. Kernel-side only."""
    global _DECLARED
    declared, _DECLARED = _DECLARED, []
    return declared


__all__ = ["ParseResult", "CROSSABLE", "DEFAULT_MAX_CHARS", "basename",
           "clean_text", "suffix",
           "max_chars", "register", "drain_registrations"]
