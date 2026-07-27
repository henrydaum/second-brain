"""What a parse hands back.

Moved out of ``plugins/services/helpers`` when the parser stopped being a
service: the kernel routes parsing, so the type it routes has to be kernel
code rather than something core reaches across the plugin boundary for.

``output`` carries the payload, and its type follows the modality:

    text      -> str (UTF-8)
    container -> list[str]                    # extracted child paths
    image     -> parser-defined image objects
    audio     -> tuple(np.ndarray, int)       # (samples, sample_rate)
    video     -> av.Container
    tabular   -> dict[sheet_name or "default", pd.DataFrame]

The first two are what the rest of the system consumes; they are simple,
universal, and they cross any boundary. The rest are *intermediates* — live
objects from a foreign library, on the way to text or to a file — and they
belong wherever the code that consumes them lives. A parser that hands one of
those across a process boundary cannot be sandboxed; one that resolves it to
text or paths inside its own box can.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("ParseResult")

# Payloads that are the *result* of parsing rather than an intermediate step,
# and therefore the only ones that can leave the box that produced them.
CROSSABLE = frozenset({"text", "container"})


@dataclass
class ParseResult:
    """One file, parsed into a standard shape."""

    modality: str = "unknown"
    success: bool = True
    error: str = ""

    # The standardized payload; see the module docstring for its type.
    output: Any = None

    # Lightweight, always populated.
    metadata: dict = field(default_factory=dict)

    # Multi-modal discovery — what else is in this file? e.g. ["image",
    # "tabular"] for a PDF with charts and photos. This is how one parse tells
    # the pipeline there is another route out of the same file.
    also_contains: list[str] = field(default_factory=list)

    @staticmethod
    def failed(error: str, modality: str = "unknown") -> "ParseResult":
        """Convenience constructor for parse failures."""
        return ParseResult(success=False, error=error, modality=modality)

    @property
    def crossable(self) -> bool:
        """Whether this payload can leave the process that produced it."""
        return self.modality in CROSSABLE
