"""
PARSER TEMPLATE
===============
A parser teaches Second Brain how to read one or more file extensions. It is a
library discovered by filename, not a plugin: there is no base class, name, or
run method.

Before writing: read docs/SDK.md, then this entire template. For details not
defined here, inspect sandbox/guest/parsing.py (ParseResult, helpers, and route
registration), parsing/registry.py (kernel discovery and routing),
parsing/kernel_sdk.py (the in-process SDK view), and sandbox/guest/loader.py
(loading a parser into a consumer's box). Validate the finished file before
relying on discovery.

  Where it goes:  DATA_DIR/workspace/parsers/parse_<format>.py
  Filename:       must start with "parse_"
  Function:       parse_<format>(sdk, path, config=None) -> ParseResult
  Registration:   register(extensions, modality, function)

THE ONE SIGNATURE HAS TWO CALLERS
---------------------------------
The kernel may call a parser to obtain a crossable result, or another sandbox
box may load the parser beside code that consumes a live result. In both cases
the parser receives `sdk`; use it for every effect. Never import kernel modules
such as `parsing`, `paths`, or `runtime`, because a subprocess cannot see them.
Import the shared contract only from `guest.parsing`.

RESULTS AND INTERMEDIATES
-------------------------
`text` and `container` results can cross the process boundary. Image, audio,
video, and tabular values are live intermediates and must be consumed inside
the same box. A task or tool requests that arrangement with
`parse_modalities = [...]`; docs/SDK.md explains the consumer side.

Use `also_contains` when one file exposes additional modalities. A container
parser returns extracted child paths and uses modality `container`.

REGISTRATION IS A DECLARATION
-----------------------------
Call `register` at module scope with literal extensions, a modality, and the
function. For an extension with several modalities, registration order chooses
the default. Extensions include the leading dot and should be lower-case.

Heavy libraries belong in `dependencies_pip` and make the parser subprocessed.
Import them inside the parse function when possible so discovery remains cheap.
Foreign libraries perform their own effects outside the Request system, which
is why isolation tightens; it does not grant additional permission.
"""

from guest.parsing import ParseResult, clean_text, max_chars, register


def parse_example(sdk, path, config=None):
    """Parse a UTF-8 .example file into bounded, normalized text."""
    try:
        text = sdk.fs.read(path)[:max_chars(config)]
        text = clean_text(text, preserve_indent=False)
        return ParseResult(
            modality="text",
            output=text,
            metadata={"char_count": len(text)},
        )
    except Exception as exc:
        sdk.log(f"Failed to parse {path}: {exc}", level="debug")
        return ParseResult.failed(str(exc), modality="text")


register([".example"], "text", parse_example)
