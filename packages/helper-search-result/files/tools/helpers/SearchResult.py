"""
Search result dataclass.

The universal container for search results across all tools and modalities.
Both lexical and semantic search tools return lists of SearchResult, and
the hybrid search tool consumes them directly for RRF fusion.

Every result carries universal fields (path, score, source, stream, modality)
plus nullable modality-specific fields that are set when applicable.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SearchResult:
    # --- Universal (always set) ---
    """Search result."""
    path: str               # file path
    score: float            # higher = better
    source: str             # provenance: "extracted", "ocr", "text_embedding", "image_embedding"
    stream: str             # retrieval method: "lexical", "text_semantic", "image_semantic"
    modality: str           # file modality: "text", "image", "audio", "video", "tabular"

    # --- Content (set when available) ---
    content: Optional[str] = None   # text snippet, OCR text, transcript, etc.

    # --- Modality-specific indices (nullable) ---
    chunk_index: Optional[int] = None   # text chunks
    image_index: Optional[int] = None   # images within a document
    # timestamp: Optional[float] = None   # audio/video position
    # row_index: Optional[int] = None     # tabular data
    # sheet_name: Optional[str] = None    # spreadsheet tab

    def to_dict(self) -> dict:
        """Handle to dict."""
        return asdict(self)
