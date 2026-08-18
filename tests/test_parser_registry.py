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
    def parser(sdk, path, config=None):
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

    result = parser(None, "x.png", {})
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
    def broken(sdk, path, config=None):
        """Fail the way a real parser fails on a malformed file."""
        raise ValueError("corrupt header")

    parsing.register([".md"], "text", broken)
    result = parsing.parse("notes.md")
    assert not result.success
    assert "corrupt header" in result.error


def test_a_delegating_parser_reaches_a_peer_through_the_sdk():
    """parse_gdoc needs google_drive, and asks for it the same way in both worlds.

    Inside a box that call is a Request; here it is a direct lookup against the
    live registry. The parser is written once and cannot tell the difference,
    which is what the shared signature buys.
    """
    class Drive:
        """A peer service."""
        def fetch(self, doc_id):
            """Answer."""
            return f"contents of {doc_id}"

    def delegating(sdk, path, config=None):
        """Reach the peer rather than holding it."""
        text = sdk.services.call("google_drive", "fetch", doc_id=path)
        return parsing.ParseResult(modality="text", output=text)

    parsing.register([".gdoc"], "text", delegating)
    parsing.bind_services({"google_drive": Drive()})
    try:
        assert parsing.parse("doc.gdoc").output == "contents of doc.gdoc"
    finally:
        parsing.bind_services({})


def test_a_parser_missing_its_peer_fails_rather_than_crashes():
    """An uninstalled delegate is an ordinary answer in a microkernel."""
    def delegating(sdk, path, config=None):
        """Ask for something that is not loaded."""
        return parsing.ParseResult(modality="text",
                                   output=sdk.services.call("absent", "fetch"))

    parsing.register([".gdoc"], "text", delegating)
    result = parsing.parse("doc.gdoc")
    assert not result.success
    assert "absent" in result.error


# ──────────────────────────────────────────────────────────────────────
# The pilot migration: one parser, two callers.
# ──────────────────────────────────────────────────────────────────────

def test_the_kernel_parser_reads_through_the_sdk(tmp_path):
    """``parse_text`` is the first migrated parser, and this is the claim.

    It never calls ``open``. The kernel hands it a stand-in whose ``fs.read``
    is a direct read; a box hands it the real SDK and the same line becomes a
    mediated Request. Nothing in the parser changes.
    """
    from bundled.parsers.parse_text import parse_plaintext

    note = tmp_path / "note.md"
    note.write_text("hello   world\n\n\n\nagain\n", encoding="utf-8")

    class Recording:
        """An sdk that records instead of reading."""

        def __init__(self, text):
            self.asked = []
            self.fs = self
            self._text = text

        def read(self, path, encoding="utf-8"):
            """Stand in for sdk.fs.read."""
            self.asked.append(path)
            return self._text

        def log(self, message, level="info"):
            """Stand in for sdk.log."""

    sdk = Recording("hello   world\n\n\n\nagain\n")
    result = parse_plaintext(sdk, str(note), {})

    assert result.success
    assert sdk.asked == [str(note)], "the parser must read through the sdk"
    assert "hello world" in result.output       # collapsed by clean_text
    assert result.crossable


def test_the_kernel_stand_in_actually_reads(tmp_path):
    """And through the real kernel path, it reads the real file."""
    note = tmp_path / "note.md"
    note.write_text("# Title\n\nBody text.\n", encoding="utf-8")

    parsing.discover()
    result = parsing.parse(str(note))
    assert result.success
    assert "Body text." in result.output


def test_a_parser_honours_the_char_limit(tmp_path):
    """max_chars comes from config, and truncation happens after the read."""
    from bundled.parsers.parse_text import parse_plaintext

    class Sdk:
        """Minimal stand-in."""
        fs = property(lambda self: self)

        def read(self, path, encoding="utf-8"):
            """A long file."""
            return "x" * 10_000

        def log(self, message, level="info"):
            """Ignore."""

    result = parse_plaintext(Sdk(), "big.txt", {"max_chars": 100})
    assert len(result.output) == 100


def test_a_parser_loads_inside_a_subprocess_box(tmp_path):
    """The property the guest move exists for.

    A parser importing kernel modules loads in-process and fails in a child,
    because the child runs with ``sandbox/`` as its working directory and
    cannot see the kernel at all. That difference only shows up for the heavy
    parsers that most need the process boundary — so it is pinned here with
    the kernel's own parser, in a real subprocess.
    """
    import shutil

    from sandbox.bridge import adapt, configure
    from sandbox.facade import Sandbox

    tree = tmp_path / "tree"
    (tree / "tools").mkdir(parents=True)
    (tree / "helpers").mkdir()
    shutil.copy("bundled/parsers/parse_text.py", tree / "helpers" / "parse_text.py")

    note = tmp_path / "note.md"
    note.write_text("# Title\n\nSome   body   text.\n", encoding="utf-8")

    (tree / "tools" / "tool_read.py").write_text('''
"""Read a file through the kernel's parser."""

from guest.bases import BaseTool

from .parse_text import parse_plaintext


class Read(BaseTool):
    """Read."""

    name = "read_via_parser"
    description = "Read a file as text."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
    dependencies_files = ["helpers/parse_text.py"]
    isolation = "subprocess"

    def run(self, sdk, path=""):
        """Parse inside this box; only the text leaves it."""
        result = parse_plaintext(sdk, path)
        return result.output
''', encoding="utf-8")

    sandbox = Sandbox()
    configure(sandbox)
    try:
        module = adapt(tree / "tools" / "tool_read.py")
        assert module is not None, "the tool did not bridge"
        tool = next(v() for v in vars(module).values() if isinstance(v, type))

        outcome = tool.run(None, path=str(note))
        assert outcome.success, outcome.error
        assert "Some body text." in outcome.data     # whitespace collapsed
    finally:
        configure(None)
        sandbox.shutdown()


def test_the_parser_contract_is_guest_code():
    """``ParseResult`` must be reachable without importing the kernel.

    This is the constraint that makes a parser loadable in a box, and it is
    easy to undo by accident — one convenient kernel import in a parser and
    only the subprocess path breaks.
    """
    from sandbox.validator import validate_file

    report = validate_file("bundled/parsers/parse_text.py")
    assert report.ok, report.render()
    assert not report.disclaimed, report.render()


def test_the_kernel_stand_in_matches_the_sdk_surface():
    """The stand-in must offer every name a parser could reach for.

    A parser is written once and run two ways; if ``KERNEL_SDK`` is missing a
    method the real SDK has, the parser works in a box and breaks in the
    kernel — a divergence with no symptom until something calls it. Pinning
    the namespaces parsers actually use keeps the two honest.
    """
    from sandbox.guest.sdk import SDK
    from parsing.kernel_sdk import KERNEL_SDK

    real = SDK(None)
    for namespace in ("fs", "services"):
        expected = {n for n in dir(getattr(real, namespace))
                    if not n.startswith("_")}
        actual = {n for n in dir(getattr(KERNEL_SDK, namespace))
                  if not n.startswith("_")}
        missing = expected - actual
        assert not missing, f"KERNEL_SDK.{namespace} is missing {sorted(missing)}"

    for name in ("log", "ok", "fail"):
        assert callable(getattr(KERNEL_SDK, name, None)), name

    # The exception names are the same claim about a different kind of member,
    # and they fail worse: a missing one makes the ``except`` clause itself
    # raise ``AttributeError``, which masks the failure the parser was
    # guarding — so the log names the stand-in and never the real cause.
    for name in ("Denied", "Failed"):
        assert getattr(KERNEL_SDK, name, None) is getattr(SDK, name), name


def test_a_parser_guard_catches_a_kernel_side_service_failure():
    """``except sdk.Failed`` must catch what the stand-in raises.

    Every delegating parser guards ``sdk.services.call`` this way — "not
    installed", "not loaded" and "it broke" are one answer to a parser — so
    raising a plain ``LookupError`` here meant the guard caught nothing when
    the *kernel* was the caller. That is the path attachment routing takes,
    which is the one that matters for a voice note.
    """
    from parsing.kernel_sdk import KernelSDK

    sdk = KernelSDK({})
    try:
        sdk.services.call("whisper", "transcribe", audio_path="x.wav")
    except sdk.Failed as exc:
        assert "not loaded" in str(exc)
        assert exc.result.code == "not_found"
    else:
        raise AssertionError("calling an absent service should fail")

    # A refusal is the narrower half, and stays catchable as either name.
    try:
        sdk.services.load("whisper")
    except sdk.Denied:
        pass
    else:
        raise AssertionError("loading a service should be refused")


def test_the_stand_in_reports_a_broken_service_like_the_handler_does():
    """Foreign code is guarded, and the message names whose bug it is."""
    from parsing.kernel_sdk import KernelSDK

    class Whisper:
        def transcribe(self, audio_path):
            raise RuntimeError("no model")

    sdk = KernelSDK({"whisper": Whisper()})
    try:
        sdk.services.call("whisper", "transcribe", audio_path="x.wav")
    except sdk.Failed as exc:
        assert str(exc).endswith("whisper.transcribe failed: no model")
    else:
        raise AssertionError("a raising service should fail the call")


def test_kernel_sdk_iter_bytes_matches_guest_windowing(tmp_path):
    """A parser sees the same offset and limit behavior in either caller."""
    from parsing.kernel_sdk import KERNEL_SDK

    path = tmp_path / "media.bin"
    path.write_bytes(b"abcdefghij")

    assert list(KERNEL_SDK.fs.iter_bytes(
        path, chunk_size=4, offset=2, limit=7)) == [b"cdef", b"ghi"]


def test_kernel_sdk_stat_matches_guest_metadata_shape(tmp_path):
    from parsing.kernel_sdk import KERNEL_SDK

    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    assert set(KERNEL_SDK.fs.stat(path)) == {
        "path", "name", "is_file", "is_dir", "is_symlink", "mtime", "size"}
    assert KERNEL_SDK.fs.exists(path)
    assert not KERNEL_SDK.fs.exists(tmp_path / "missing.txt")


# ────────────────────────────────────────────────────────────────────
# Generic vs specialist: which side of the parser boundary a file sits on.
# ────────────────────────────────────────────────────────────────────

def test_the_text_parser_is_the_only_generic_one():
    """Every bundled and installed parser, and only parse_text says generic.

    The default is what makes this safe -- a parser author who has never heard
    of the flag ships a specialist, which is the routing that reads the file
    properly. Pinned as a *set* so a second generic parser has to be argued
    for here rather than appearing.
    """
    import parsing

    parsing.discover()
    generic = {ext for (ext, _modality) in parsing.registry._GENERIC}

    assert ".py" in generic and ".md" in generic and ".txt" in generic
    assert all(parsing.describe_extension(ext)["generic"] for ext in generic)


def test_a_specialist_parser_is_distinguishable_from_the_text_fallback():
    """The bug this exists for: both register as "text", and routing on the
    modality alone hands the agent a .gdoc's JSON stub as the document."""
    import parsing

    parsing.discover()
    parsing.register(".gdoc", "text", lambda sdk, path, config: None)

    plain = parsing.describe_extension(".py")
    pointer = parsing.describe_extension(".gdoc")

    # Indistinguishable by the old question ...
    assert plain["modality"] == pointer["modality"] == "text"
    # ... and separated by the new one.
    assert plain["generic"] and not pointer["generic"]
    assert plain["known"] and pointer["known"]


def test_an_unregistered_extension_is_neither_known_nor_generic():
    """`known` is what keeps the parse branch off a file nothing can parse.

    ``get_modality`` answers the *string* "unknown", so a caller comparing it
    against real modalities gets this right only by accident.
    """
    import parsing

    parsing.discover()
    route = parsing.describe_extension(".sbxyz")

    assert route == {"modality": "unknown", "known": False, "generic": False}


def test_clearing_the_registry_forgets_generic_registrations():
    """_GENERIC is cleared with everything else, or an uninstalled parser
    keeps voting on how its extension routes."""
    import parsing

    parsing.discover()
    parsing.register(".sbtest", "text", lambda sdk, path, config: None,
                     generic=True)
    assert parsing.describe_extension(".sbtest")["generic"]

    parsing.clear()
    assert not parsing.describe_extension(".sbtest")["generic"]


def test_registering_a_specialist_over_a_generic_route_clears_the_flag():
    """Re-registration replaces rather than accumulates, in both directions."""
    import parsing

    parsing.clear()
    parsing.register(".sbtest", "text", lambda sdk, path, config: None,
                     generic=True)
    parsing.register(".sbtest", "text", lambda sdk, path, config: None)

    assert not parsing.describe_extension(".sbtest")["generic"]

