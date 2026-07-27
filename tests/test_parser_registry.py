"""The kernel's parser registry: routing, and what may leave a parse.

Parsing used to be a service, which forced every result to travel as a live
object and made the heaviest code in the system the one part that could not be
sandboxed. It is now kernel routing plus importable parser functions, so these
tests are about the two things the kernel actually owns: knowing what a file
is, and knowing which parser answers for it.
"""

import pytest

import parsing


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry is module-global; a leaked parser hides a real failure."""
    parsing.clear()
    yield
    parsing.clear()


def _stub(output="text", modality="text"):
    """A parser function shaped like a real parse_*.py helper."""
    def parser(path, config, services):
        """Answer with a fixed result."""
        return parsing.ParseResult(modality=modality, output=output)
    return parser


# ──────────────────────────────────────────────────────────────────────
# Routing — the half core actually depends on.
# ──────────────────────────────────────────────────────────────────────

def test_native_defaults_answer_with_no_parsers_installed():
    """A bare kernel still knows a .png is an image.

    Attachment routing leans on this: it is what lets a vision model be handed
    an image on an install with zero parser packages.
    """
    assert parsing.get_modality(".gif") == "image"
    assert parsing.get_modality(".png") == "image"
    assert parsing.get_modality(".mp3") == "audio"
    assert parsing.get_modality(".mp4") == "video"


def test_an_unknown_extension_says_so():
    """"unknown" is a routing answer, not a failure."""
    assert parsing.get_modality(".zzz") == "unknown"


def test_a_registered_parser_beats_the_native_default():
    """Installing a parser is how you change what a file is treated as."""
    assert parsing.get_modality(".png") == "image"
    parsing.register([".png"], "text", _stub())
    assert parsing.get_modality(".png") == "text"


def test_the_first_registration_sets_the_default_modality():
    """A PDF is text first and images second, and order is the declaration."""
    parsing.register([".pdf"], "text", _stub())
    parsing.register([".pdf"], "image", _stub(modality="image"))

    assert parsing.get_modality(".pdf") == "text"
    assert sorted(parsing.get_modalities_for(".pdf")) == ["image", "text"]


def test_extensions_normalize():
    """Callers are inconsistent about dots and case; the registry is not."""
    parsing.register(["PDF"], "text", _stub())
    assert parsing.get_modality(".pdf") == "text"
    assert parsing.get_modality("pdf") == "text"
    assert ".pdf" in parsing.get_supported_extensions()


def test_clear_keeps_the_native_defaults():
    """They are what the kernel knows, not what happens to be installed."""
    parsing.register([".png"], "text", _stub())
    parsing.clear()
    assert parsing.get_modality(".png") == "image"
    assert parsing.get_supported_extensions() == set()


# ──────────────────────────────────────────────────────────────────────
# Parsers as libraries.
# ──────────────────────────────────────────────────────────────────────

def test_a_parser_can_be_looked_up_and_called_directly():
    """The importable half: this is how a box parses without anything crossing.

    Code needing a heavy modality takes the function and calls it in its own
    process, so the PIL image or open container never has to travel.
    """
    parsing.register([".png"], "image", _stub(output="an image object",
                                              modality="image"))
    parser = parsing.parser_for(".png", "image")
    assert parser is not None

    result = parser("x.png", {}, {})
    assert result.output == "an image object"
    assert not result.crossable      # exactly why it was called in-box


def test_looking_up_a_parser_that_is_not_there_returns_none():
    """Callers branch on absence; they should not have to catch."""
    assert parsing.parser_for(".png", "image") is None


def test_text_and_paths_are_the_crossable_results():
    """The line between a parse result and a parse intermediate."""
    assert parsing.ParseResult(modality="text").crossable
    assert parsing.ParseResult(modality="container").crossable
    for modality in ("image", "audio", "video", "tabular"):
        assert not parsing.ParseResult(modality=modality).crossable


# ──────────────────────────────────────────────────────────────────────
# Parsing in this process.
# ──────────────────────────────────────────────────────────────────────

def test_parse_dispatches_and_defaults_the_modality():
    """No modality given means "whatever this extension is"."""
    parsing.register([".md"], "text", _stub(output="hello"))
    assert parsing.parse("notes.md").output == "hello"


def test_parse_reports_a_missing_parser_rather_than_raising():
    """A missing parser package is an ordinary answer in a microkernel."""
    result = parsing.parse("notes.md", "text")
    assert not result.success
    assert "No parser" in result.error


def test_parse_reports_an_unroutable_file():
    """Nothing registered and no native default: say why, do not guess."""
    result = parsing.parse("notes.zzz")
    assert result.modality == "unknown"
    assert "No parser registered" in result.metadata["reason"]


def test_a_raising_parser_becomes_a_failed_result():
    """One bad parser must not take down whatever asked it to parse."""
    def broken(path, config, services):
        """Fail the way a real parser fails on a malformed file."""
        raise ValueError("corrupt header")

    parsing.register([".md"], "text", broken)
    result = parsing.parse("notes.md")
    assert not result.success
    assert "corrupt header" in result.error


def test_bound_services_reach_delegating_parsers():
    """parse_gdoc needs google_drive; that is what binding replaced.

    The service dict was the one thing ParserService did that the registry
    could not, and it turned out to be a reference rather than a lifecycle.
    """
    seen = {}

    def delegating(path, config, services):
        """Record what peers were available."""
        seen["services"] = services
        return parsing.ParseResult(modality="text", output="ok")

    parsing.register([".gdoc"], "text", delegating)
    parsing.bind_services({"google_drive": "a service"})
    try:
        parsing.parse("doc.gdoc")
        assert seen["services"] == {"google_drive": "a service"}
    finally:
        parsing.bind_services({})
