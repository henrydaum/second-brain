"""Undo a shell escape that reached a tool which is not a shell.

The default data directory is ``/data/Second Brain``, so most absolute paths
in this system contain a space. A model that has been writing shell commands
learns to escape it — ``/data/Second\\ Brain/...`` — and then carries the habit
into the *file* tools, where nothing ever removes the backslash again. The
path is looked up literally, no such file exists, and the error quotes a name
the model believes it typed correctly.

It is a habit rather than a slip, which is what makes it worth repairing:
in the benchmark corpus one task produced twelve consecutive ``read_file``
failures this way, every one of them the same escaped root, plus a ``grep``
that never searched. Nothing recovered until the task ended.

**Only a root this system already knows is repaired, and only as literal
text.** That is the same rule ``tool_run_command._quote_known_roots`` follows
in the opposite direction, and it is what makes the repair safe rather than a
guess: a backslash is a legal character in a POSIX filename, so un-escaping
blindly would invent a different path. Matching the escaped spelling of a
known root cannot — the root is known to exist under its real name, and no
file is legitimately called ``/data/Second\\ Brain``.

Everything else is left exactly as written, including a backslash anywhere
below the root, which stays guesswork and stays untouched.
"""


def _roots(sdk):
    """The absolute roots worth repairing: the project, and the data tree."""
    found = []
    for name in ("project", "data"):
        try:
            root = sdk.paths.get(name)
        except Exception:                                # noqa: BLE001
            continue
        if root and " " in str(root):
            found.append(str(root))
    # Longest first: the data directory may sit inside the project root, and
    # the outer match is the one worth repairing.
    return sorted(found, key=len, reverse=True)


def unescape_known_roots(sdk, raw):
    """``(path, repaired root or None)`` — the path with a known root unescaped.

    Answers the original string untouched when nothing matches, so a caller
    can pass every path through this without deciding first whether it looks
    escaped.
    """
    text = str(raw or "")
    if "\\" not in text:
        return text, None
    for root in _roots(sdk):
        escaped = root.replace(" ", "\\ ")
        if escaped != root and text.startswith(escaped):
            return root + text[len(escaped):], root
    return text, None


def note(root):
    """What to tell the model, so the habit does not survive the repair.

    Disclosed rather than repaired silently: a tool that quietly accepts a
    path different from the one it was handed teaches nothing, and the next
    call arrives escaped too.
    """
    if not root:
        return ""
    return (f" (read as {root!r} — a backslash before a space is shell "
            "quoting, and file tools take the plain path)")
