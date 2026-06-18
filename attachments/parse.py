"""Build an :class:`Attachment` for a file on disk via the parser service.

This is the attachment system's single integration point with parsing. There
is no separate attachment-parser registry anymore: modality detection and the
LLM-readable text rendering both come from the unified parser service
(``services["parser"]``), so anything the pipeline can parse, an attachment
can too — and installing a parser package lights up both at once.

Routing happens later in ``AttachmentBundle.split_for_llm(capabilities)``:
    1. Native  - the LLM has the capability for this modality -> raw path inlined.
    2. Parsed  - the parser produced a text blurb              -> appended as suffix.
    3. Pointer - neither                                       -> just file name + path.
"""

from __future__ import annotations

import logging
from pathlib import Path

from attachments.attachment import Attachment

logger = logging.getLogger("AttachmentParse")


def parse_attachment(
    path: str,
    *,
    file_name: str | None = None,
    services: dict | None = None,
    config: dict | None = None,
) -> Attachment:
    """Build an :class:`Attachment`, using the parser service for modality and
    an LLM-readable text rendering.

    The file's modality comes from ``parser.get_modality`` (which knows native
    image/audio/video types even when no heavy parser is installed). A text
    blurb is produced when a ``(extension, "text")`` parser exists — text-class
    files (txt/csv/pdf/docx/…) yield their contents; native-modality files
    usually have no text parser and fall through to native routing or the
    pointer fallback at LLM time.
    """
    p = Path(path)
    ext = p.suffix.lower()
    name = file_name or p.name or "attachment"
    services = services or {}
    config = config or {}

    parser = services.get("parser")
    modality = "binary"
    parsed_text: str | None = None
    metadata: dict = {}

    if parser is not None:
        try:
            modality = parser.get_modality(ext) or "binary"
        except Exception:
            modality = "binary"
        try:
            result = parser.parse(str(p), "text", config=config)
        except Exception as e:
            logger.warning(f"Attachment text parse for {ext} failed on {p.name}: {e}")
            result = None
        output = getattr(result, "output", None)
        if isinstance(output, str) and output.strip():
            parsed_text = output
            metadata = dict(getattr(result, "metadata", {}) or {})

    if modality in (None, "unknown"):
        modality = "binary"

    return Attachment(
        path=str(p),
        extension=ext,
        file_name=name,
        modality=modality,
        parsed_text=parsed_text,
        metadata=metadata,
    )
