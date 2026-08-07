"""
SCRIPT TEMPLATE
===============
Not every piece of sandboxed code is a plugin. A script is the other shape: a
file of functions that take `sdk`, with no base class and nothing that
registers it. Reference for authoring one; not imported by the running system.

It is not quite true that a script declares nothing: `box` and `timeout` are
read from module scope, and both are covered below.

Use a script for one-off computation, scratch work, analysis, or anything the
kernel does not need to register and schedule. If nothing has to *find* your
code by name, it does not need a plugin class.

Read docs/SDK.md for the Request surface. This file covers what is specific to
scripts and helper files.

Before writing: read docs/SDK.md, then this entire template. For details not
defined here, inspect sandbox/guest/sdk.py (`sdk.scripts` and other call
signatures), sandbox/guest/loader.py (entry loading and box lifetime),
sandbox/isolation.py (script recognition), and sandbox/policy.py (permission
classification). Validate the finished file before running it.

  Where it goes:  <DATA_DIR>/workspace/scripts/, and nowhere else.
                  `sdk.paths.get("scripts")` will tell you the absolute path.
                  The directory is the entire declaration — a script has no
                  prefix, no base class and no keyword that could say what it
                  is — so a file anywhere else is REFUSED rather than asked
                  about, and you have left a stray file behind. Spell the whole
                  path out when you write it and again when you run it.
  Filename:       anything EXCEPT a family prefix. A file named tool_*.py,
                  task_*.py, service_*.py, command_*.py or frontend_*.py must
                  contain a matching plugin class — the validator will reject
                  it otherwise. Name scripts anything else.
  Entry point:    any function you name when running it; `main` by default

How it gets run:

    sdk.scripts.run(path)                       # calls main(sdk)
    sdk.scripts.run(path, "summarize", path="notes.md")

Whatever the function returns comes back. Pass `wait=False` for work that
should not hold up a turn. From outside the sandbox it is the `run_script`
tool, which takes the same absolute path.

Reach for a script instead of a shell command. Both do work on this machine,
but a shell command is a process the kernel cannot see into, so it asks the
user every single time; a script is contained — every effect inside it arrives
at the gate on its own — so it asks nothing at all. The one exception is a
script importing a library outside the standard library: that is asked about
once per run, and the library is named.


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
early exit; shutdown is the kernel's decision, not yours.


HOW LONG YOU GET
----------------
60 seconds by default. Declare more at module scope, the same way you declare
`box`:

    timeout = 600

Two things about that number are worth knowing before you pick one.

It measures *running* time, not elapsed time. Waiting on the kernel does not
count against it — a script blocked for four minutes inside `sdk.proc.run` or
`sdk.agent.spawn` is charged nothing for the wait. What it measures is your own
computation, and it accumulates across the whole run, so a long loop doing a
little work per iteration is what actually breaches it.

It is a request, not a grant: the kernel clamps at 600s, so declaring 5000 gets
you 600. And a separate wall-clock ceiling of 600s bounds every run however it
spends the time, blocked or not. So ten minutes of elapsed time is the real
limit on a script, and work that needs longer than that wants to be a task.

Ask `sdk.budget()` what is actually left, rather than guessing from the number
you declared — it answers `{running, wall, deadline, ceiling}` in seconds, and
`deadline` is what the kernel is enforcing after the clamp. Checking it is what
lets a long loop stop itself:

    def sweep(sdk, documents):
        done = []
        for doc in documents:
            if sdk.budget()["running"] < 20:
                break
            done.append(analyse(sdk, doc))
        return sdk.ok({"done": done, "resume_at": len(done)})

Worth doing in any loop over an unknown amount of work. The alternative is not
"it runs a bit longer" — it is the watchdog killing the box, so a run
three-quarters of the way through returns nothing at all. Calling it costs
nothing: it is read-only, so it draws no dialog and writes no ledger row.


DOING SEVERAL THINGS AT ONCE
----------------------------
`wait=False` hands back an id, and that is how a script fans work out:

    ids = [sdk.scripts.run(path, "analyse", wait=False, doc=d)["id"]
           for d in documents]
    for report in sdk.scripts.collect(ids):
        sdk.log(report["state"], report["data"])

Each one is a box of its own, so they genuinely run in parallel. `collect`
answers with `id`, `script`, `state`, `ok`, `data` and `error` per run;
`timeout=0` polls without waiting and leaves anything still running collectable;
`sdk.scripts.stop(id)` cancels one. Every report is delivered once.

Reach for this when the work is code, and for `sdk.agent.spawn` when it needs
judgement — that one costs a model call per child and this one costs none.


TRYING AGAIN
------------
`sdk.retry(fn)` retries only what the kernel said was worth retrying, which it
knows because the handler that failed set it:

    page = sdk.retry(lambda: sdk.net.http(url))

A locked file and a timed-out request come back retryable; a malformed query
does not, and neither does a refusal — `sdk.Denied` propagates on the first
attempt however you configure it, because asking again is a second dialog in
front of somebody who already said no. The backoff sleeps, and sleeping is
running time, so a retry loop inside a long run is one more reason to watch
`sdk.budget()`.


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

    Watch the ordering. `Denied` is a subclass of `Failed`, so `except
    sdk.Failed` catches refusals too — put `except sdk.Denied` first whenever
    "the user said no" deserves different handling from "the disk is full".
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
