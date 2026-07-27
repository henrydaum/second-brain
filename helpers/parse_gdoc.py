"""Google Docs parser (.gdoc shortcut files).

A ``.gdoc`` file is a small JSON shortcut holding a Drive ``doc_id``. Parsing
it means fetching the document body from Google Drive, so this parser
delegates to the ``google_drive`` peer service. When that service isn't loaded
(it ships as part of the Google Drive package), the parse fails cleanly and
the caller falls back to a pointer.
"""


dependencies_files = ['services/service_drive.py']
dependencies_pip = []

import json

from guest.parsing import ParseResult, clean_text, max_chars, register


def parse_gdoc(sdk, path: str, config: dict = None) -> ParseResult:
    """Parse a .gdoc shortcut and fetch its content from Google Drive.

    Requires the ``google_drive`` service to be loaded. Reaching it through
    ``sdk.services`` rather than holding the instance is what lets this file
    run in a box: the call becomes a Request, and the service stays wherever
    it lives.
    """
    if not sdk.services.list().get("google_drive"):
        return ParseResult.failed(
            "Drive service not loaded — retry after loading",
            modality="text",
        )

    try:
        gdoc_data = json.loads(sdk.fs.read(path))

        doc_id = gdoc_data.get("doc_id")
        if not doc_id:
            return ParseResult.failed("No doc_id found in .gdoc file", modality="text")

        # The service handles the API call and thread safety internally.
        content = sdk.services.call("google_drive", "download_text", doc_id=doc_id)
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
        sdk.log(f"Failed to parse gdoc {path}: {e}", level="error")
        return ParseResult.failed(str(e), modality="text")


register(".gdoc", "text", parse_gdoc)
