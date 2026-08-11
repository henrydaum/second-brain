"""One message, however many files.

A person who attaches three files and types a line has sent **one message**.
The kernel could always carry it — ``ConversationLoop.drive`` bundles the whole
of ``cs.pending_attachments`` into the first model call of the turn — but the
wire could only say one file at a time, and a ``send_attachment`` hands
priority straight to the agent. So the second file arrived at a session that
was already busy and was answered "Still working. Send /cancel to interrupt.",
which is what made a transport with a file picker a one-file transport.

The fix is an argument (``files``) rather than a type, and these pin the two
halves of it: the handler that prepares a submit, and the one action that
queues everything it carried.
"""

from types import SimpleNamespace

import pytest

from sandbox.handlers.kernel import _frontend_submit, _prepare_attachment
from state_machine.action_map import ACTION_SEND_ATTACHMENT
from state_machine.conversation import ConversationState, Participant


def _files(tmp_path, *names):
    """Real files on disk, since ingestion reads them."""
    made = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        made.append(str(path))
    return made


# ──────────────────────────────────────────────────────────────────────
# Preparing the submit.
# ──────────────────────────────────────────────────────────────────────

def test_several_files_become_one_action_payload(tmp_path):
    one, two = _files(tmp_path, "chart.png", "notes.pdf")

    prepared = _prepare_attachment(None, {
        "files": [{"path": one, "file_name": "chart.png"},
                  {"path": two, "file_name": "notes.pdf"}],
        "caption": "what do these have in common?",
    })

    assert [f["file_name"] for f in prepared["files"]] == ["chart.png",
                                                           "notes.pdf"]
    assert [f["extension"] for f in prepared["files"]] == ["png", "pdf"]


def test_the_messages_caption_rides_on_the_first_file_only(tmp_path):
    """It is the line the person typed, not a label on each file.

    Copied onto all three it would be written to history three times and read
    to the model three times, which is how a one-line question becomes a
    stutter.
    """
    one, two, three = _files(tmp_path, "a.png", "b.png", "c.png")

    prepared = _prepare_attachment(None, {
        "files": [{"path": one}, {"path": two}, {"path": three}],
        "caption": "which is sharpest?",
    })

    assert [f["caption"] for f in prepared["files"]] == [
        "which is sharpest?", "", ""]


def test_a_file_may_still_state_its_own_caption(tmp_path):
    one, two = _files(tmp_path, "a.png", "b.png")

    prepared = _prepare_attachment(None, {
        "files": [{"path": one, "caption": "before"},
                  {"path": two, "caption": "after"}],
        "caption": "message level",
    })

    assert [f["caption"] for f in prepared["files"]] == ["before", "after"]


def test_one_file_in_a_list_is_the_flat_form_exactly(tmp_path):
    """The widening has nothing for an existing frontend to migrate to."""
    one, = _files(tmp_path, "chart.png")

    listed = _prepare_attachment(None, {
        "files": [{"path": one, "file_name": "chart.png", "caption": "hi"}]})
    flat = _prepare_attachment(None, {
        "path": one, "file_name": "chart.png", "caption": "hi"})

    assert listed == flat
    assert "files" not in listed


def test_a_list_carrying_no_path_is_a_coded_failure(tmp_path):
    result = _prepare_attachment(None, {"files": [{"file_name": "ghost.png"}]})

    assert not result.ok
    assert result.code == "invalid_argument"


def test_one_pathless_file_refuses_the_whole_message(tmp_path):
    """Skipping it would send a message missing a file nobody is told about."""
    one, = _files(tmp_path, "a.png")

    result = _prepare_attachment(None, {
        "files": [{"path": one}, {"file_name": "ghost.png"}]})

    assert not result.ok
    assert result.code == "invalid_argument"


def test_the_plain_path_is_still_plain(tmp_path):
    """No ``files``, no metadata, no ingest: the original path, untouched."""
    one, = _files(tmp_path, "chart.png")

    assert _prepare_attachment(None, {"path": one}) is None


# ──────────────────────────────────────────────────────────────────────
# Submitting it.
# ──────────────────────────────────────────────────────────────────────

class _Adapter:
    """A frontend adapter that records what the state machine was handed."""

    name = "http"
    background_submit = False

    def __init__(self):
        self.submitted = []

    def submit(self, session_key, action_type, payload=None):
        self.submitted.append((session_key, action_type, payload))
        return SimpleNamespace(ok=True)

    def submit_attachment(self, session_key, path, extension=None):
        self.submitted.append((session_key, "plain", path))
        return SimpleNamespace(ok=True)


def test_a_multi_file_submit_enacts_exactly_one_action(tmp_path, monkeypatch):
    """The whole point: one action, so only one turn is started.

    Two submits would be two ``send_attachment`` actions, and the second meets
    a busy session — which is the bug, not an implementation detail.
    """
    from sandbox.handlers import kernel

    adapter = _Adapter()
    monkeypatch.setattr(kernel, "_at_desk", lambda args: (adapter, None))
    one, two = _files(tmp_path, "a.png", "b.pdf")

    result = _frontend_submit(None, {
        "session_key": "http:main", "input_kind": "attachment",
        "files": [{"path": one}, {"path": two}], "caption": "read these"})

    assert result.ok
    assert len(adapter.submitted) == 1
    key, action_type, payload = adapter.submitted[0]
    assert (key, action_type) == ("http:main", ACTION_SEND_ATTACHMENT)
    assert len(payload["files"]) == 2


# ──────────────────────────────────────────────────────────────────────
# Enacting it.
# ──────────────────────────────────────────────────────────────────────

def _state(**kwargs):
    """A state machine whose parser answers like the runtime's does."""
    from attachments.attachment import Attachment

    def parse(content):
        name = content.get("file_name") or "file"
        caption = content.get("caption") or ""
        pointer = f"[Attached image file: {name}]"
        text = f"{caption}\n\n{pointer}".strip() if caption else pointer
        return {**content, "text": text, "attachment": Attachment(
            path=content.get("path") or "", extension="png", file_name=name,
            modality="image")}

    return ConversationState(
        [Participant("user", "user"), Participant("agent", "agent")],
        turn_priority="user", phase="awaiting_input",
        attachment_parser=parse, **kwargs)


def test_every_file_is_queued_by_the_one_action():
    cs = _state()

    result = cs.enact(ACTION_SEND_ATTACHMENT, {"files": [
        {"path": "/tmp/a.png", "file_name": "a.png", "caption": "these two"},
        {"path": "/tmp/b.png", "file_name": "b.png"},
    ]}, "user")

    assert result.ok
    assert [a.file_name for a in cs.pending_attachments] == ["a.png", "b.png"]
    # And the agent has the turn, once.
    assert cs.turn_priority == "agent"


def test_the_message_writes_one_history_row():
    """``dispatch.text_after_action`` reads ``parsed['text']`` for history.

    A row per file would be three user messages for one thing the person did.
    """
    cs = _state()

    result = cs.enact(ACTION_SEND_ATTACHMENT, {"files": [
        {"path": "/tmp/a.png", "file_name": "a.png", "caption": "these two"},
        {"path": "/tmp/b.png", "file_name": "b.png"},
    ]}, "user")

    text = (result.data or {})["parsed"]["text"]
    assert text == ("these two\n\n[Attached image file: a.png]\n"
                    "[Attached image file: b.png]")


def test_one_refused_extension_queues_nothing():
    """Every file is checked before any is parsed.

    Half a message is worse than none: the person is told it landed and the
    agent answers about whichever files happened to pass.
    """
    cs = _state(allowed_attachment_extensions=["png"])

    result = cs.enact(ACTION_SEND_ATTACHMENT, {"files": [
        {"path": "/tmp/a.png", "file_name": "a.png"},
        {"path": "/tmp/b.exe", "file_name": "b.exe"},
    ]}, "user")

    assert not result.ok
    assert cs.pending_attachments == []
    assert cs.turn_priority == "user"


def test_a_single_file_action_is_unchanged():
    cs = _state()

    result = cs.enact(ACTION_SEND_ATTACHMENT,
                      {"path": "/tmp/a.png", "file_name": "a.png",
                       "caption": "look"}, "user")

    assert result.ok
    assert [a.file_name for a in cs.pending_attachments] == ["a.png"]
    parsed = (result.data or {})["parsed"]
    assert parsed["text"] == "look\n\n[Attached image file: a.png]"
    assert "files" not in parsed


# ──────────────────────────────────────────────────────────────────────
# All the way to the model.
# ──────────────────────────────────────────────────────────────────────

def test_the_turn_hands_the_model_every_queued_file():
    """The half that already worked, pinned so it keeps working.

    ``drive`` bundles whatever is pending into the first call — this is what
    the wire was failing to fill.
    """
    from attachments.attachment import AttachmentBundle

    cs = _state()
    cs.enact(ACTION_SEND_ATTACHMENT, {"files": [
        {"path": "/tmp/a.png", "file_name": "a.png"},
        {"path": "/tmp/b.png", "file_name": "b.png"},
    ]}, "user")

    bundle = AttachmentBundle.from_iterable(cs.pending_attachments)

    assert len(list(bundle)) == 2


if __name__ == "__main__":       # pragma: no cover
    pytest.main([__file__])
