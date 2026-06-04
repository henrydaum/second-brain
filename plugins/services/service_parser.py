"""
Parser service.

Wraps the parser registry as a standard service so callers access parsing
through the uniform ``context.services.get("parser").parse(...)`` pattern.

The service lifecycle triggers parser discovery: _load() rebuilds the
registry by scanning every ``services/helpers/parse_*.py`` module across the
built-in, sandbox, and installed plugin roots and importing each, which fires
its top-level ``parser_registry.register(...)`` calls. This is the kernel's
parser-package substrate: a parser package ships a ``services/helpers/
parse_*.py`` helper (a non-entrypoint payload) and it lights up the next time
the parser service loads; uninstalling removes the file and a reload drops it.

The live service registry is injected through BaseService so parsers can
delegate to peers (e.g. parse_gdoc -> google_drive, parse_audio -> whisper).
"""

import logging

from plugins.BaseService import BaseService, EXTENSION
from plugins.services.helpers import parser_registry

logger = logging.getLogger("ParserService")


class ParserService(BaseService):
    """File parser dispatch. Discovers and registers parsers on load."""

    model_name = "parser"
    shared = True
    lifecycle = EXTENSION
    config_settings: list = []

    def _load(self) -> bool:
        """Rebuild the parser registry from the discovered helper modules."""
        self._discover_parsers()
        self.loaded = True
        return True

    def _discover_parsers(self) -> None:
        """Clear the registry and (re)import every parse_*.py helper module.

        Scans the ``services/helpers`` directory under each plugin root in
        precedence order (built-in first, so the kernel's text parser always
        wins). Each module registers its (extension, modality) mappings on
        import. Heavy parsers degrade on their own via lazy ImportError
        guards, so a missing optional dependency just leaves those extensions
        unregistered rather than failing the scan.
        """
        from plugins.helpers.plugin_paths import plugin_dirs
        from plugins.plugin_discovery import _load_plugin_module

        parser_registry.clear()
        seen: set[str] = set()
        count = 0
        for service_dir in plugin_dirs("service"):
            helpers = service_dir.path / "helpers"
            if not helpers.exists():
                continue
            for py_file in sorted(helpers.glob("parse_*.py")):
                if py_file.stem in seen:
                    continue  # earlier (higher-precedence) root wins
                module_name = f"{service_dir.root.module}.services.helpers.{py_file.stem}"
                module = _load_plugin_module(
                    module_name, py_file, service_dir.root.built_in, reload=True
                )
                if module is not None:
                    seen.add(py_file.stem)
                    count += 1
        logger.info(
            f"Parser discovery: {count} parser module(s), "
            f"{len(parser_registry.get_supported_extensions())} extension(s)."
        )

    def get_modality(self, extension: str) -> str:
        """Default modality for a file extension. See parser_registry.get_modality."""
        return parser_registry.get_modality(extension)

    def parse(self, path: str, modality: str = None, config: dict = None):
        """Parse a file and return a ParseResult. See parser_registry.parse."""
        return parser_registry.parse(path, modality, config, self.services)


def build_services(config: dict) -> dict:
    """Build services."""
    return {"parser": ParserService()}
