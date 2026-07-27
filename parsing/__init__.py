"""Parsing — the kernel's file-type authority.

Import from here rather than from the submodules::

    from parsing import get_modality, parse, ParseResult

Parsing is *routing plus a library*, not a service. The kernel owns the
routing (which parsers exist, what modality an extension is) because that is
standing knowledge every part of the system needs and none of it should load.
The parsers themselves are ordinary importable functions, so code that needs a
heavy modality pulls the parser into its own box and consumes the result
there — which is the only way a foreign library can be sandboxed at all.
"""

from .registry import (bind_services, clear, discover, get_modalities_for,
                       get_modality, get_supported_extensions, parse,
                       parser_for, register)
from .result import CROSSABLE, ParseResult
from .utils import clean_text, max_chars

__all__ = ["ParseResult", "CROSSABLE", "bind_services", "clean_text", "clear",
           "discover", "get_modalities_for", "get_modality",
           "get_supported_extensions", "max_chars", "parse", "parser_for",
           "register"]
