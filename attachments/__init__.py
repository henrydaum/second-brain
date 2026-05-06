"""Attachment dataclasses, lightweight per-extension parsers, and the
attachment cache (where frontends drop incoming files).

Attachment parsers are intentionally NOT plugins. Each one is a small
function in ``attachments/parsers/parser_*.py`` that turns a file into a
text blurb (e.g. audio -> transcription). The blurb is stored on the
``Attachment`` dataclass and used by the LLM service as a prompt suffix
when the model lacks the native capability for that modality.

Three-tier routing in ``AttachmentBundle.for_llm(capabilities)``:
    1. Native    - capability matches modality -> raw file path inlined.
    2. Parsed    - parser produced text       -> appended as a suffix.
    3. Pointer   - neither                    -> just file name + path.
"""

from attachments.attachment import Attachment, AttachmentBundle
from attachments.registry import parse_attachment, modality_for, register
from attachments.cache import save

# Importing the parsers package triggers each parser's ``register(...)``
# call so the registry is populated as soon as anything imports this
# module.
from attachments import parsers  # noqa: F401

__all__ = [
    "Attachment",
    "AttachmentBundle",
    "parse_attachment",
    "modality_for",
    "register",
    "save",
]
