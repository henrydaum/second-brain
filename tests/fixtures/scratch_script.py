"""An arbitrary script — the case with no plugin contract at all.

This is what an agent writes for a one-off computation: no base class, no
name, no declarations. A file with functions that take ``sdk``.
"""


def summarize(sdk, path):
    """Read a file and report some statistics about it."""
    lines = sdk.fs.read(path).splitlines()
    return {
        "lines": len(lines),
        "words": sum(len(line.split()) for line in lines),
        "longest": max((len(line) for line in lines), default=0),
    }


def pure_math(sdk, values):
    """Compute without asking for anything at all."""
    return sum(values) / len(values) if values else 0
