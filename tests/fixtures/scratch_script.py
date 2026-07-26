"""An arbitrary script — the case with no plugin contract at all.

This is what an agent writes for a one-off computation: no base class, no
name, no family, no declarations. It is a file with functions that take
``sdk``, and the sandbox runs it exactly like anything else.
"""


def summarize(sdk, path):
    """Read a file and report some statistics about it."""
    r = sdk.fs.read(path)
    if not r:
        return sdk.fail(r.error)
    lines = r.data.splitlines()
    return sdk.ok({
        "lines": len(lines),
        "words": sum(len(line.split()) for line in lines),
        "longest": max((len(line) for line in lines), default=0),
    })


def pure_math(sdk, values):
    """Compute without asking for anything at all."""
    return sdk.ok(sum(values) / len(values) if values else 0)
