"""The parser registry — the kernel's answer to "what kind of file is this?"

Parsing used to be a service, and that was a mistake worth naming. A service
is a *runtime object other code calls*, which forced every parse result to
travel as a live object: a PIL image, a numpy array, an open ``av.Container``.
Those cannot cross a process boundary, so nothing that parsed could ever be
sandboxed, and the heaviest, least trustworthy code in the system — foreign
C libraries chewing on arbitrary files — was the one part that had to run
unmediated in the kernel's own process.

So the authority moved back into the kernel and the shape changed with it:

- **The kernel routes.** ``get_modality``, ``get_supported_extensions`` and
  ``discover`` are standing knowledge about file types, not a capability
  anyone loads. They answer in strings and belong nowhere else.
- **Parsers are libraries, not endpoints.** A ``parse_*.py`` helper is
  imported by whatever consumes it. Code needing decoded audio puts the
  parser in *its own box* alongside the thing that transcribes it, so the
  waveform never crosses anything and the transcript does.
- **What leaves a parse is text or paths.** Everything else is an
  intermediate; see :mod:`parsing.result`.

Discovery still works the way it did — a package ships a
``helpers/parse_*.py`` and it lights up on the next scan — because
that part was never the problem. What changed is who owns the scan.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .kernel_sdk import KERNEL_SDK
from sandbox.guest.parsing import ParseResult, drain_registrations

logger = logging.getLogger("Parsing")


# ===================================================================
# THE REGISTRY
#
# Key:   (extension, modality)  e.g. (".pdf", "text"), (".pdf", "image")
# Value: parser function  ->  func(sdk, path, config) -> ParseResult
#
# _MODALITY_MAP holds the default modality per extension, set by register():
# the first modality registered for an extension becomes its default.
# ===================================================================

_REGISTRY: dict[tuple[str, str], callable] = {}
_MODALITY_MAP: dict[str, str] = {}


# Native modalities the LLM may ingest directly even when no parser is
# installed for the extension. The kernel's standing knowledge of "what kind
# of file is this?" — independent of which heavy parsers happen to be
# installed — so attachment routing can still inline e.g. a .png into a vision
# model with zero image parsers present. Registered parsers take precedence.
_NATIVE_DEFAULTS: dict[str, str] = {
    # Image
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".tiff": "image", ".tif": "image",
    ".heic": "image", ".heif": "image", ".ico": "image",
    # Audio
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".m4a": "audio",
    ".aac": "audio", ".ogg": "audio", ".oga": "audio", ".opus": "audio",
    ".wma": "audio",
    # Video
    ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video",
    ".avi": "video",
}


def _normalize(extension: str) -> str:
    """Lower-cased, dot-prefixed. Callers are inconsistent; this is not."""
    ext = extension or ""
    return ext.lower() if ext.startswith(".") else f".{ext.lower()}"


def clear():
    """Empty the registry so a fresh scan can rebuild it.

    The native-modality defaults are static and are not cleared: they are what
    the kernel knows regardless of what is installed.
    """
    _REGISTRY.clear()
    _MODALITY_MAP.clear()


def register(extensions: str | list[str], modality: str, func: callable):
    """Register a parser for one or more extensions under a modality.

    Called at import time by each ``parse_*.py`` helper. The first modality
    registered for an extension becomes its default.
    """
    if isinstance(extensions, str):
        extensions = [extensions]
    for extension in extensions:
        ext = _normalize(extension)
        _REGISTRY[(ext, modality)] = func
        if ext not in _MODALITY_MAP:
            _MODALITY_MAP[ext] = modality


def bind_services(services: dict) -> None:
    """Point the kernel-side sdk's service lookup at the live registry.

    For parsers that delegate (parse_gdoc -> google_drive, parse_audio ->
    whisper). A reference, not a lifecycle: parsing is not a thing that loads.
    Inside a box the same call goes through ``sdk.services.call`` as a Request
    instead, which is the whole point of the shared signature.
    """
    KERNEL_SDK.services.bind(services)


# ===================================================================
# ROUTING — what core actually uses
# ===================================================================

def get_modality(extension: str) -> str:
    """The default modality for an extension.

    A registered parser wins; otherwise the static native defaults; otherwise
    "unknown".
    """
    ext = _normalize(extension)
    return _MODALITY_MAP.get(ext) or _NATIVE_DEFAULTS.get(ext) or "unknown"


def get_modalities_for(extension: str) -> list[str]:
    """Every registered modality for an extension."""
    ext = _normalize(extension)
    return [modality for (e, modality) in _REGISTRY if e == ext]


def get_supported_extensions() -> set[str]:
    """Every extension with at least one registered parser."""
    return {ext for ext, _ in _REGISTRY}


def parser_for(extension: str, modality: str):
    """The parser function for one (extension, modality) pair, or None.

    The importable half of the registry: code that wants to parse *inside its
    own box* asks for the function and calls it directly, so whatever heavy
    object comes back never has to travel.
    """
    return _REGISTRY.get((_normalize(extension), modality))


# ===================================================================
# PARSING — for callers in this process
# ===================================================================

def parse(path: str, modality: str = None, config: dict = None,
          sdk=None) -> ParseResult:
    """Parse a file in *this* process and return a ParseResult.

    Fine for text and for extracted paths, which is what core asks for. For
    the heavier modalities prefer :func:`parser_for` from inside the box that
    consumes the result — calling through here puts a live PIL image or an
    open container in the caller's process, which is exactly what stops it
    being sandboxable.

    ``sdk`` defaults to the kernel stand-in; a box passes its real one, and
    the parser cannot tell the difference.
    """
    config = config or {}
    path_obj = Path(path)
    extension = path_obj.suffix.lower()

    if modality is None:
        modality = get_modality(extension)
        if modality == "unknown":
            return ParseResult(
                modality="unknown",
                metadata={"reason": f"No parser registered for {extension}"})

    parser_func = _REGISTRY.get((extension, modality))
    if parser_func is None:
        return ParseResult.failed(
            error=f"No parser for ({extension}, {modality})", modality=modality)

    logger.debug("Parsing %r as %s (ext=%s)", path_obj.name, modality, extension)
    try:
        return parser_func(KERNEL_SDK if sdk is None else sdk, path, config)
    except Exception as exc:
        logger.error("Parser failed for %s as %s: %s", path_obj.name, modality,
                     exc)
        return ParseResult.failed(error=str(exc), modality=modality)


# ===================================================================
# DISCOVERY — finding the installed parsers
# ===================================================================

def discover() -> int:
    """Rebuild the registry by importing every installed parser helper.

    Scans ``parsers/parse_*.py`` at each tree's root in precedence order
    (bundled first, so the kernel's text parser always wins) and imports each,
    firing its module-level ``register(...)`` calls. Heavy parsers guard their
    own imports, so a missing optional dependency leaves those extensions
    unregistered rather than failing the scan.

    Installing a parser package and calling this makes it live; uninstalling
    and calling this drops it. That is the whole lifecycle, and it is a
    function rather than a service because nothing about it needs to persist.
    """
    import trees
    from plugins.plugin_discovery import _load_plugin_module

    clear()
    seen: set[str] = set()
    count = 0
    for root, parsers in trees.dirs_for("parsers"):
        if not parsers.exists():
            continue
        for py_file in sorted(parsers.glob("parse_*.py")):
            if py_file.stem in seen:
                continue          # an earlier, higher-precedence root won
            module_name = f"{root.module}.parsers.{py_file.stem}"
            drain_registrations()          # discard anything left by a failure
            module = _load_plugin_module(module_name, py_file, root.builtin,
                                         reload=True)
            if module is None:
                continue
            # The parser declared itself into the guest-side collector on
            # import; this is where those declarations become the kernel's
            # registry. Draining per module keeps one broken parser from
            # stealing another's registrations.
            for extensions, modality, func in drain_registrations():
                register(extensions, modality, func)
            seen.add(py_file.stem)
            count += 1

    logger.info("Parser discovery: %d parser module(s), %d extension(s).",
                count, len(get_supported_extensions()))
    return count
