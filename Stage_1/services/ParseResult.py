import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("ParseResult")

@dataclass
class ParseResult:
    """
    output type by modality:
        text      -> str (UTF-8)
        image     -> list[PIL.Image.Image]
        audio     -> tuple(np.ndarray, int)  # (samples, sample_rate)
        video     -> av.Container
        tabular   -> dict[sheet_name or "default", pd.DataFrame]
        container -> list[str]  # extracted child paths
    """
    modality: str = "unknown"
    success: bool = True
    error: str = ""

    # The standardized payload
    output: Any = None  # The raw output of the parser, in one of the standard formats above.

    # Metadata — lightweight, always populated
    metadata: dict = field(default_factory=dict)

    # Multi-modal discovery — what else is in this file?
    also_contains: list[str] = field(default_factory=list)
    # e.g. ["image", "tabular"] for a PDF with charts and photos

    @staticmethod
    def failed(error: str, modality: str = "unknown") -> "ParseResult":
        """Convenience constructor for parse failures."""
        return ParseResult(success=False, error=error, modality=modality)
