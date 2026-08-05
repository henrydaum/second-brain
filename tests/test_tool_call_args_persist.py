"""Do a tool call's arguments survive into ``conversation_messages``?

The memory bundle's curator decides which of its two jobs it has by asking
whether the agent *opened* a note that had been surfaced to it. The offer is
recorded by ``service_memory``; the open is a ``read_file`` call, and the only
place that is observable after the fact is the stored transcript. So the whole
loop rests on one question this file answers: does the path the agent passed to
a tool appear in the row that gets saved, in a form something can match on?

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
                     tool_call_id=None, tool_name=None):
        self.rows.append({"conversation_id": conversation_id, "role": role,
                          "content": content, "tool_call_id": tool_call_id,
                          "tool_name": tool_name})


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
