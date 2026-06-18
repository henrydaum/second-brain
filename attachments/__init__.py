"""Attachment dataclasses, the parser-service-backed attachment builder, and
the attachment cache (where frontends drop incoming files).

Attachments no longer carry their own parser registry. ``parse_attachment``
builds an :class:`Attachment` using the unified parser service
(``services["parser"]``): the parser supplies both the file's modality and an
LLM-readable text blurb. ``AttachmentBundle.split_for_llm(capabilities)`` then does
three-tier routing:
    1. Native  - capability + backend support match -> passed as Attachment.
    2. Parsed  - parser produced text       -> appended as a suffix.
    3. Pointer - neither                     -> just file name + path.
"""

from attachments.attachment import Attachment, AttachmentBundle
from attachments.parse import parse_attachment
from attachments.cache import save

__all__ = [
    "Attachment",
    "AttachmentBundle",
    "parse_attachment",
    "save",
]
