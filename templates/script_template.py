"""
SCRIPT TEMPLATE
===============
Not every piece of sandboxed code is a plugin. A script is the other shape: a
file of functions that take `sdk`, with no base class and no declarations.
Reference for authoring one; not imported by the running system.

Use a script for one-off computation, scratch work, analysis, or anything the
kernel does not need to register and schedule. If nothing has to *find* your
code by name, it does not need a plugin class.

Read docs/SDK.md for the Request surface. This file covers what is specific to
scripts and helper files.

  Where it goes:  anywhere the caller can point at
  Filename:       anything EXCEPT a family prefix. A file named tool_*.py,
                  task_*.py, service_*.py, command_*.py or frontend_*.py must
                  contain a matching plugin class — the validator will reject
                  it otherwise. Name scripts anything else.
  Entry point:    any function you name when running it

How it gets run:

    from sandbox.facade import Sandbox

    sandbox = Sandbox(context=ctx, runtime=runtime)
    result = sandbox.run("scratch/summarize.py", "summarize",
                         kwargs={"path": "notes.md"})

`run()` blocks and returns a Result. `start()` returns immediately with a Run
you can wait on or cancel — the wait=False shape, for work that should not
hold up a turn.


THE SAME RULES APPLY
--------------------
Being a script does not relax anything. The validator reads the file, the gate
classifies every Request, and unsafe work still asks the user. What you save is
the ceremony, not the boundary.

Everything pure runs normally and costs nothing: arithmetic, strings, json,
re, math, datetime, collections, itertools, hashlib, statistics, dataclasses.
The test for what needs a Request is always the same — does it touch disk,
network, clock, or process?


NOTHING SURVIVES BETWEEN RUNS
-----------------------------
An ephemeral box is torn down when the work finishes, so module-level state is
discarded after each call. Do not cache anything at module scope and expect to
find it later. If you need something kept, you need a service or a persistent
box, not a global.

A script cannot end itself except by returning. `sdk.respond(value)` is an
early exit; timeouts and shutdown are the kernel's decision, not yours.


HELPER FILES AND BOXES
----------------------
A box is one execution context: one process, one memory space, one lifetime.
Files in the same box import each other normally. Files in different boxes
cannot reach each other at all — the only way across is a Request.

Declaring nothing gets you a box of your own, named after the file. To group a
script with its helpers, give every file the same box and use relative imports:

    # helper_words.py
    box = "wordcount"

    def count_words(text):
        return len(text.split())

    # summarize.py
    box = "wordcount"
    from .helper_words import count_words

Relative imports matter: they are what lets a file move between the built-in,
sandbox, and installed trees without edits.

Resolution is most-restrictive-wins. Joining a box can only ever narrow what
you may do, so a careless file cannot loosen a box by moving into it.
"""


def summarize(sdk, path):
    """Count the lines and words in a file.

    A complete sandbox program. One Request, plain data back.
    """
    lines = sdk.fs.read(path).splitlines()
    return {"lines": len(lines), "words": sum(len(line.split()) for line in lines)}


def compare(sdk, first, second):
    """Report which of two files is denser, mixing Requests with pure code."""
    # Two Requests...
    a = summarize(sdk, first)
    b = summarize(sdk, second)

    # ...and then ordinary Python, which costs nothing and asks no one.
    def density(stats):
        """Average words per line, guarding the empty case."""
        return stats["words"] / stats["lines"] if stats["lines"] else 0.0

    winner = first if density(a) >= density(b) else second
    return {"denser": winner, first: density(a), second: density(b)}


def audit(sdk, root="."):
    """Walk a tree and summarize what is in it.

    Shows the failure idiom: catch only what you can act on. A file that
    cannot be read is worth skipping; a denied listing is worth reporting.
    """
    try:
        paths = sdk.fs.list(root, pattern="*.md")
    except sdk.Denied:
        return sdk.fail(f"Not allowed to list {root}.")

    total, skipped = 0, []
    for path in paths:
        try:
            total += len(sdk.fs.read(path).split())
        except sdk.Failed:
            # One unreadable file should not sink the whole audit.
            skipped.append(path)

    return sdk.ok(
        {"files": len(paths), "words": total, "skipped": skipped},
        llm_summary=f"{len(paths)} files, {total} words, {len(skipped)} unreadable.",
    )
