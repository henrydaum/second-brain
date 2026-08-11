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
        attachment = Attachment(path=content.get("path") or "",
                                extension="png", file_name=name,
                                modality="image")
        return {**content, "text": content.get("caption") or "",
                "attachment": attachment, "record": attachment.record()}

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
    """``dispatch`` reads the text and the records off one result.

    A row per file would be three user messages for one thing the person did,
    and the text is the person's line — once, not once per photo.
    """
    cs = _state()

    result = cs.enact(ACTION_SEND_ATTACHMENT, {"files": [
        {"path": "/tmp/a.png", "file_name": "a.png", "caption": "these two"},
        {"path": "/tmp/b.png", "file_name": "b.png"},
    ]}, "user")

    assert (result.data or {})["parsed"]["text"] == "these two"
    assert [r["file_name"] for r in result.data["records"]] == ["a.png",
                                                                "b.png"]


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
    assert parsed["text"] == "look"
    assert "files" not in parsed
    assert [r["file_name"] for r in result.data["records"]] == ["a.png"]


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


# ──────────────────────────────────────────────────────────────────────
# The column. A file is a thing, not a sentence about a thing — it used to
# be written into the message text as "[Attached image file: x.png (cached
# at …)]", so the one row that says a file arrived said it in prose: a
# client could only get it back by parsing English, and a person who typed
# those characters was indistinguishable from a file.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """A real database, since the point is what a column does."""
    from pipeline.database import Database

    return Database(str(tmp_path / "test.db"))


def _record(name="chart.png", path="/tmp/chart.png", modality="image"):
    return {"path": path, "file_name": name, "modality": modality,
            "extension": ".png"}


def test_the_text_is_what_the_person_typed(db):
    cid = db.create_conversation("Main")
    db.save_message(cid, "user", "what is this?", attachments=[_record()])

    row = db.get_conversation_messages(cid)[0]

    assert row["content"] == "what is this?"
    assert row["attachments"] == [_record()]


def test_a_message_with_no_files_reads_back_as_an_empty_list(db):
    """Not None, not "[]" — one shape for every reader."""
    cid = db.create_conversation("Main")
    db.save_message(cid, "user", "just talking")

    assert db.get_conversation_messages(cid)[0]["attachments"] == []


def test_the_records_survive_a_whole_conversation_rewrite(db):
    """``iterate_agent_turn`` rewrites every row from the live history.

    A key dropped there is one that survives until the next background turn
    and then does not, which is the worst kind of loss: it needs a background
    turn to reproduce.
    """
    cid = db.create_conversation("Main")
    db.replace_conversation_messages(cid, [
        {"role": "user", "content": "what is this?",
         "attachments": [_record()]},
        {"role": "assistant", "content": "A chart."},
    ])

    rows = db.get_conversation_messages(cid)

    assert [r["attachments"] for r in rows] == [[_record()], []]


def test_history_carries_the_records_and_not_the_prose(db):
    """The row -> provider-history half of the round trip."""
    from state_machine.serialization import messages_to_history

    cid = db.create_conversation("Main")
    db.save_message(cid, "user", "what is this?", attachments=[_record()])

    history = messages_to_history(db.get_conversation_messages(cid))

    assert history == [{"role": "user", "content": "what is this?",
                        "attachments": [_record()]}]


def test_the_model_still_reads_one_message(db):
    """Rendered at call time, which is the only place that knows it is a model.

    The pointer line is deliberately byte-identical to what used to be welded
    into the text: every conversation written before the column still has
    those lines in its content, and the model must not meet two spellings of
    one thing.
    """
    from attachments.attachment import with_pointers

    assert with_pointers("what is this?", [_record()]) == (
        "what is this?\n\n"
        "[Attached image file: chart.png (cached at /tmp/chart.png)]")


def test_an_uncaptioned_file_is_still_a_message():
    """The guard used to be ``text`` alone, and worked only by accident.

    A caption-less photo had non-empty text *because* the pointer was welded
    into it. With the files in their own column that text is empty, so a guard
    reading "did the person say anything" drops the only record that a file
    was ever sent.
    """
    from runtime.dispatch import absorb_user_action
    from state_machine.errors import ActionResult

    session = SimpleNamespace(key="chat", history=[], conversation_id=None)
    runtime = SimpleNamespace(db=None, config={})
    result = ActionResult(True, "send_attachment",
                          data={"parsed": {"text": ""},
                                "records": [_record()]})

    absorb_user_action(runtime, session, "send_attachment", "", result)

    assert session.history == [{"role": "user", "content": "",
                                "attachments": [_record()]}]


def test_no_provider_ever_sees_the_key():
    """``messages`` goes to a provider API verbatim.

    A field no schema knows is either rejected outright or silently believed,
    so the rendering step has to drop it — this is the assertion that keeps
    the column from reaching an HTTP request.
    """
    from runtime.conversation_loop import _for_provider

    rendered = _for_provider({"role": "user", "content": "what is this?",
                              "attachments": [_record()]})

    assert "attachments" not in rendered
    assert rendered["content"].endswith(
        "[Attached image file: chart.png (cached at /tmp/chart.png)]")


def test_the_whole_chain_from_a_submit_to_the_model(tmp_path):
    """Every piece at once, because the pieces are the risk.

    A person attaches a file and types a line. The row keeps them apart; the
    model is handed them together; the message the provider receives carries
    no key the API has never heard of.
    """
    import state_machine  # noqa: F401  - settles the package-init cycle

    from tests.support import make_runtime, response

    photo = tmp_path / "chart.png"
    photo.write_bytes(b"\x89PNG\r\n\x1a\n")

    runtime, session, llm = make_runtime(
        tmp_path, [response(content="A chart.")], name="chain.db")
    out = runtime.handle_action("s", "send_attachment", {
        "path": str(photo), "file_name": "chart.png",
        "caption": "what is this?"})
    assert out.ok

    row = next(r for r in runtime.db.get_conversation_messages(
        session.conversation_id) if r["role"] == "user")
    assert row["content"] == "what is this?"
    assert [r["file_name"] for r in row["attachments"]] == ["chart.png"]

    sent = llm.calls[0]
    user = next(m for m in reversed(sent) if m["role"] == "user")
    assert "[Attached image file: chart.png" in user["content"]
    assert not any("attachments" in m for m in sent)
    # And the file itself went natively, which is the other half of what an
    # attachment is for.
    assert [a["file_name"] for a in llm.attachments[0]] == ["chart.png"]


def test_a_conversation_written_before_the_column_is_left_alone(db):
    """Nothing rewrites a message somebody sent.

    Old rows keep the pointer in their content and have no record, which is
    exactly how they have always read — including to the model, which is why
    no backfill guesses at where prose ends and a file begins.
    """
    from state_machine.serialization import messages_to_history

    cid = db.create_conversation("Main")
    old = ("what is this?\n\n[Attached image file: chart.png "
           "(cached at /tmp/chart.png)]")
    db.save_message(cid, "user", old)

    history = messages_to_history(db.get_conversation_messages(cid))

    assert history == [{"role": "user", "content": old}]


if __name__ == "__main__":       # pragma: no cover
    pytest.main([__file__])
