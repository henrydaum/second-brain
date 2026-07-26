"""A persistent script with no plugin contract at all — a scratchpad server.

The module *is* the object: its functions are the methods, and its globals are
the state that survives between calls. This is what an agent gets when it
wants somewhere to keep working notes across a conversation, without any of
the plugin machinery.

Note it is persistent because the kernel opened a persistent box for it, not
because of anything written here. Nothing in this file could make it so.
"""

box = "scratchpad"

_notes = {}
_calls = 0


def remember(sdk, key, value):
    """Store a note."""
    global _calls
    _calls += 1
    _notes[key] = value
    return sdk.ok(len(_notes))


def recall(sdk, key):
    """Fetch a note, or None."""
    global _calls
    _calls += 1
    return sdk.ok(_notes.get(key))


def stats(sdk):
    """Report how much state has accumulated."""
    return sdk.ok({"notes": len(_notes), "calls": _calls})
