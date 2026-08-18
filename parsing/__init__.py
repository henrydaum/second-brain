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

from .registry import (bind_services, clear, describe_extension, discover,
                       get_modalities_for, get_modality,
                       get_supported_extensions, parse, parser_for, register,
                       sources_for)
# The contract itself lives in the guest, because a parser is guest code and
# the child process cannot see the kernel. Kernel callers reach it from here
# so they need not care where it physically lives.
from sandbox.guest.parsing import (CROSSABLE, DEFAULT_MAX_CHARS, ParseResult,
                                   basename, suffix,
                                   clean_text, max_chars)

__all__ = ["ParseResult", "CROSSABLE", "DEFAULT_MAX_CHARS",
           "basename", "bind_services", "clean_text", "clear",
           "describe_extension", "suffix",
           "discover", "get_modalities_for", "get_modality",
           "get_supported_extensions", "max_chars", "parse", "parser_for",
           "register", "sources_for"]
