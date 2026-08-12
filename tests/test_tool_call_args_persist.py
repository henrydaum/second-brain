"""Do a tool call's arguments survive into ``conversation_messages``?

Anything reading a conversation back — the memory curator's transcript, an
audit, a replay — sees only what ``save_history_message`` wrote. So the
question this file answers is whether the arguments the agent passed to a tool
appear in that row at all, in a form something can match on.

It was written for a loop that no longer exists: the memory bundle used to
infer that a note had been *opened* by scanning the transcript for a
``read_file`` call and comparing the path it named. That inference is gone —
``tool_memory`` records the fact when it happens — and the property
survives it, because the transcript is still the only place a past tool call
is observable and the encoding is still the trap described below.

``save_history_message`` packs an assistant message's ``tool_calls`` with
``json.dumps``, and the provider format carries ``arguments`` as a JSON
*string* — so the arguments are encoded twice. That is fine for a POSIX path
and emphatically not fine for a Windows one, which is the case this file
exists to pin.
"""

import json

from state_machine.serialization import messages_to_history, save_history_message


class _DB:
    """Just enough database to see what a save would have written."""

    def __init__(self):
        self.rows = []

    def save_message(self, conversation_id, role, content,
                     tool_call_id=None, tool_name=None, attachments=None,
                     author=None):
        self.rows.append({"conversation_id": conversation_id, "role": role,
                          "content": content, "tool_call_id": tool_call_id,
                          "tool_name": tool_name,
                          "attachments": attachments or [],
                          "author": author})


def _saved(path):
    """The stored row for an assistant turn that read ``path``."""
    db = _DB()
    save_history_message(db, 1, {
        "role": "assistant",
        "content": "Let me check what I noted before.",
        "tool_calls": [{"id": "call_1", "name": "read_file",
                        "arguments": json.dumps({"path": path})}],
    })
    return db.rows[0]


def test_a_posix_path_survives_verbatim():
    """The Mac case, and the one the curator's matching relies on."""
    path = "/Users/henry/Library/Second Brain/workspace/memory/commit_trailers.md"

    row = _saved(path)

    assert row["role"] == "assistant"
    assert path in row["content"], row["content"]


def test_a_windows_path_does_not_survive_verbatim():
    """Double encoding, and the reason matching cannot be a bare substring.

    ``arguments`` is already a JSON string when it arrives, so its backslashes
    are escaped once; ``save_history_message`` then ``json.dumps`` the whole
    structure and escapes them again. A literal ``in`` test against the path
    the service logged therefore fails on Windows while passing on macOS — a
    platform-dependent silent miss, which is the worst shape this bug could
    have taken.
    """
    path = r"Z:\Second Brain\workspace\memory\commit_trailers.md"

    row = _saved(path)

    assert path not in row["content"]
    assert path.replace("\\", "\\\\\\\\") in row["content"], row["content"]


def test_the_filename_survives_on_both_platforms():
    """Double encoding does not alter ordinary filename characters."""
    for path in ("/home/h/workspace/memory/commit_trailers.md",
                 r"Z:\Second Brain\workspace\memory\commit_trailers.md"):
        assert "commit_trailers.md" in _saved(path)["content"], path


def test_the_arguments_are_recoverable_not_merely_present():
    """A reader that wants the value, rather than a substring, can have it."""
    path = "/home/h/workspace/memory/commit_trailers.md"

    packed = json.loads(_saved(path)["content"])
    call = packed["tool_calls"][0]

    assert call["name"] == "read_file"
    assert json.loads(call["arguments"])["path"] == path


def test_the_round_trip_restores_the_call_for_the_next_turn():
    """The stored row is also what the provider gets back, so it has to hold."""
    path = "/home/h/workspace/memory/commit_trailers.md"

    history = messages_to_history([_saved(path)])

    assert history[0]["tool_calls"][0]["name"] == "read_file"
    assert path in history[0]["tool_calls"][0]["arguments"]
