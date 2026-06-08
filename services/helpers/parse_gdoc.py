"""Google Docs parser (.gdoc shortcut files).

A ``.gdoc`` file is a small JSON shortcut holding a Drive ``doc_id``. Parsing
it means fetching the document body from Google Drive, so this parser
delegates to the ``google_drive`` peer service. When that service isn't loaded
(it ships as part of the Google Drive package), the parse fails cleanly and
the caller falls back to a pointer.
"""


dependencies_files = ['services/service_drive.py']

import json
import logging
from pathlib import Path

from plugins.services.helpers.ParseResult import ParseResult
from plugins.services.helpers import parser_registry as registry
from plugins.services.helpers.parsing_utils import clean_text, max_chars

logger = logging.getLogger("ParseGDoc")


def parse_gdoc(path: str, config: dict, services: dict = None) -> ParseResult:
    """Parse a .gdoc shortcut and fetch its content from Google Drive.

    Requires the ``google_drive`` service to be loaded.
    """
    drive_svc = services.get("google_drive") if services else None

    if drive_svc is None or not getattr(drive_svc, "loaded", False):
        return ParseResult.failed(
            "Drive service not loaded — retry after loading",
            modality="text",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            gdoc_data = json.load(f)

        doc_id = gdoc_data.get("doc_id")
        if not doc_id:
            return ParseResult.failed("No doc_id found in .gdoc file", modality="text")

        # Service handles the API call and thread safety internally.
        content = drive_svc.download_text(doc_id)
        if content is None:
            return ParseResult.failed("Failed to download document", modality="text")

        limit = max_chars(config)
        content = clean_text(content[:limit])

        return ParseResult(
            modality="text",
            output=content,
            metadata={
                "char_count": len(content),
                "source": "google_drive",
                "doc_id": doc_id,
            },
        )
    except Exception as e:
        logger.error(f"Failed to parse gdoc {Path(path).name}: {e}")
        return ParseResult.failed(str(e), modality="text")


registry.register(".gdoc", "text", parse_gdoc)
