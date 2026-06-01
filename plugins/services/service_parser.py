"""
Parser service.

Wraps the parser registry as a standard service so callers access parsing
through the uniform ``context.services.get("parser").parse(...)`` pattern.

The service lifecycle triggers parser discovery: _load() imports parser
modules under plugins/services/helpers/, and each module registers
its (extension, modality) -> function mappings on import via
parser_registry.register().

The live service registry is injected through BaseService so parsers can
delegate to peers (e.g. parse_gdoc -> google_drive, parse_audio -> whisper).
"""

import logging

from plugins.BaseService import BaseService
from plugins.services.helpers import parser_registry
# Lite kernel: only the text parser ships built-in. Importing it triggers its
# top-level registry.register() calls for text/code/PDF/DOCX/PPTX extensions.
# Richer modality parsers (image, audio, video, tabular, container) live in the
# store and re-register themselves when their plugin is installed.
from plugins.services.helpers import parse_text

logger = logging.getLogger("ParserService")


class ParserService(BaseService):
    """File parser dispatch. Registers built-in parsers on load."""

    model_name = "parser"
    shared = True
    config_settings: list = []

    def _load(self) -> bool:
        # The text parser registers its extensions on import (module top-level).
        # Nothing further to load for the lite kernel.
        """Internal helper to load parser service."""
        self.loaded = True
        return True

    def unload(self):
        # The registry itself is just a dict of callables — no heavyweight
        # resources to release. Leave the registrations in place so a
        # subsequent load() is idempotent (Python won't re-run module-level
        # register() calls on re-import).
        """Handle unload."""
        self.loaded = False

    def parse(self, path: str, modality: str = None, config: dict = None):
        """Parse a file and return a ParseResult. See parser_registry.parse."""
        return parser_registry.parse(path, modality, config, self.services)


def build_services(config: dict) -> dict:
    """Build services."""
    return {"parser": ParserService()}
