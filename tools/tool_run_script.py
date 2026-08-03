"""Run a script the agent wrote — SDK code, contained, no dialog.

This exists because of what section 18 of the security contract concludes: a
command line cannot be classified, so every shell command is asked about, and
that left the agent's *cheapest* capability also its most dangerous one. Under
any pressure to get work done, everything routes through the one door meant to
be hardest.

A script is the door that should have been there instead. It is Python over the
SDK, so there is nothing to interpret — every effect inside it arrives at the
kernel's gate individually and is judged there, with the script still in the
chain. Running one therefore widens nothing, and costs no interruption.

The tool itself is a translation layer over one Request, the same shape as
``validate``: ``script.run`` resolves the path, validates the bytes, opens a
subprocess box and calls the entry function. Nothing here decides whether a
script may run — that is ``sandbox/policy.py``, which is the whole point of
authorization not living in the code being authorized.
"""

dependencies_files = []
dependencies_pip = []
requests = ["script.run", "fs.list", "fs.delete", "paths.get"]

from guest.bases import BaseTool

# A returned value is shown to the model in full up to this, then summarized.
# A script that produces more than this wanted to write a file.
MAX_RESULT_CHARS = 4000

# How many scripts the prompt names. Past this the listing stops helping for a
# reason tokens do not capture: a bare filename says nothing about what the
# script does, so a long list is noise rather than a menu, and the entries
# worth re-running are the recent ones. The rest stay one fs.list away.
MAX_LISTED = 20


class RunScript(BaseTool):
    """Run script."""
    name = "run_script"
    description = (
        "Run a script you wrote in the scripts/ directory. A script is a file of "
        "SDK code with a main(sdk) function — no base class, no declarations. "
        "Prefer this over run_command for anything expressible in Python: a "
        "script is contained and runs without interrupting the user, while every "
        "shell command asks for permission. Validate the file first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute path to the script. It must already live in the "
                    "scripts directory named in the Scripts section of your "
                    "instructions — a script anywhere else is refused."
                ),
            },
            "entry": {
                "type": "string",
                "description": "Function to call. Defaults to main.",
            },
            "args": {
                "type": "object",
                "description": "Keyword arguments passed to the entry function.",
            },
            "delete_after": {
                "type": "boolean",
                "description": (
                    "Delete the script once it has run. Use for genuinely "
                    "one-off work; leave it off for anything worth keeping and "
                    "improving."
                ),
            },
        },
        "required": ["path"],
    }
    requires_services = []

    def agent_prompt(self, sdk) -> str:
        """Where scripts go and why to reach for one."""
        scripts = sdk.paths.get("scripts")
        sep = "\\" if "\\" in scripts else "/"
        return (
            f"""## Scripts
Write a script when you need to *do* something rather than build a capability: read a hundred files and summarize them, reshape some data, call an API in a loop, clean up a directory. Write it into the scripts directory and run it with run_script.

**Write it to this exact directory, by absolute path:**

    {scripts}

So a script called `tidy.py` is written to `{scripts}{sep}tidy.py` — spell that whole path out when you create the file and again when you call run_script. A relative filename lands wherever the process happens to be sitting, which is the project root, not here; that is not the scripts directory, so the run is *refused* rather than asked about, and you have left a stray file behind. Nothing else about the file says it is a script — the directory is the entire declaration.

A script is a plain file with functions that take `sdk`:

    def main(sdk, limit=10):
        rows = sdk.db.query("SELECT title FROM conversations LIMIT ?", [limit])
        return [r["title"] for r in rows]

Whatever `main` returns comes back to you. Call validate(path=...) first — a script that does not conform will be refused rather than run.

Reach for this instead of run_command. Both end up doing work on this machine, but a shell command is a process the kernel cannot see into, so it asks the user every single time; a script is contained, so it does not ask at all. The only script that interrupts anyone is one importing a library outside the standard library, and then the user is told which library and why. Keep to the standard library and the `sdk` and you will never be stopped.

Scripts persist, and that is useful — improve one across conversations rather than rewriting it each time. Pass delete_after=true only when the work is genuinely single-use.

## Scripts you have
These sit in the directory above — join it to the name to get the path run_script wants. Most recently changed first.

{_existing(sdk, scripts)}"""
        )

    def run(self, sdk, **kwargs):
        """Run script."""
        path = (kwargs.get("path") or "").strip()
        if not path:
            return sdk.fail("path is required.")

        entry = (kwargs.get("entry") or "main").strip()
        args = kwargs.get("args") or {}
        if not isinstance(args, dict):
            return sdk.fail("args must be an object of keyword arguments.")

        try:
            value = sdk.scripts.run(path, entry, **args)
        except sdk.Denied as refused:
            # Worth distinguishing from a breakage: the user said no, and the
            # answer is to explain or to write the script differently, not to
            # retry the same call.
            return sdk.fail(f"Not permitted: {refused}")
        except sdk.Failed as failed:
            # The script is left on disk whatever delete_after said. A failed
            # run is the case where the source is most worth having.
            return sdk.fail(f"{sdk.path.name(path)} failed: {failed.error}")

        removed = ""
        if kwargs.get("delete_after"):
            removed = _remove(sdk, path)

        return sdk.ok(value, llm_summary=_summarize(sdk, path, value, removed))


def _remove(sdk, path) -> str:
    """Delete a script that has served its purpose, and say what happened.

    Deleting inside the agent's own tree is not asked about, for the same
    reason writing there is not. Anywhere else it is, and a refusal here is
    reported rather than raised — the script already ran, and losing that
    result over a failed cleanup would be the wrong trade.
    """
    try:
        sdk.fs.delete(path)
        return f"{sdk.path.name(path)} was deleted after running."
    except sdk.Failed as failed:
        return f"Could not delete {sdk.path.name(path)}: {failed.error}"


def _summarize(sdk, path, value, removed: str) -> str:
    """What the model is told, when the raw value is the wrong thing to show."""
    name = sdk.path.name(path)
    body = "" if value is None else str(value)
    if len(body) > MAX_RESULT_CHARS:
        body = (sdk.text.truncate(body, MAX_RESULT_CHARS)
                + f"\n\n[{len(body)} characters; truncated. Have the script "
                  f"write its output to a file if you need all of it.]")
    lines = [f"{name} ran." if not body else f"{name} returned:\n\n{body}"]
    if removed:
        lines.append(removed)
    return "\n\n".join(lines)


def _existing(sdk, scripts) -> str:
    """The most recent scripts in the scripts directory, by name.

    Listed in the prompt because the cheapest useful move is often to run one
    that already exists, and a script written three conversations ago is
    invisible otherwise. Bounded because this block is *dynamic*: written as a
    method, it lands in the live context update rather than the cached prefix,
    so it is paid on every model call and an unbounded directory listing grows
    without ever being cached.

    Names, not paths. The directory is stated verbatim a few lines above, so a
    full absolute path per entry repeated it once per script — about six times
    the characters to say the same thing.

    Sorting happens kernel-side (``sort="mtime"``) but the cap does not: the
    count of what was left out is worth more than the round trip it saves, and
    the answer's own ``truncated`` flag is a bool that cannot supply it.
    """
    try:
        answer = sdk.fs.list(scripts, pattern="*.py", files_only=True,
                             sort="mtime")
    except Exception:
        # Never break the prompt over a listing. A missing directory is the
        # ordinary case on a fresh install, not an error worth surfacing here.
        return "None yet."

    names = [sdk.path.name(entry) for entry in (answer or {}).get("entries", [])]
    found = [name for name in names if not name.startswith("_")]
    if not found:
        return "None yet."

    shown = found[:MAX_LISTED]
    lines = [f"  {name}" for name in shown]
    if len(found) > len(shown):
        lines.append(f"  ({len(found) - len(shown)} older not shown — "
                     f"list the directory yourself to see them all.)")
    return "\n".join(lines)
