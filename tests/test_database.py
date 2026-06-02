"""Tests for the SQLite layer (``pipeline.database.Database``).

The database backs two core kernel concerns: the file/task pipeline queue and
durable conversation storage. These tests run against a fresh on-disk DB in a
temp dir, so the schema bootstrap in ``_setup`` is exercised for real.
"""

import pytest

from pipeline.database import Database, DEFAULT_USER_ID


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    yield database


# ── Files ────────────────────────────────────────────────────────────

def test_upsert_and_list_files(db):
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 100.0)
    db.upsert_file("/notes/b.md", "b.md", ".md", "text", 200.0)

    assert db.get_all_files() == {"/notes/a.md": 100.0, "/notes/b.md": 200.0}
    assert db.get_files_by_modality("text") == ["/notes/a.md", "/notes/b.md"]


def test_upsert_is_idempotent_and_updates_mtime(db):
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 100.0)
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 150.0)

    files = db.get_all_files()
    assert files == {"/notes/a.md": 150.0}


def test_remove_file_also_clears_its_tasks(db):
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 100.0)
    db.enqueue_task("/notes/a.md", "extract_text")
    db.remove_file("/notes/a.md")

    assert db.get_all_files() == {}
    assert db.get_pending_tasks("extract_text") == []


# ── Task queue ───────────────────────────────────────────────────────

def test_enqueue_claim_complete_lifecycle(db):
    db.enqueue_task("/notes/a.md", "extract_text")
    assert not db.is_task_done("/notes/a.md", "extract_text")

    claimed = db.claim_tasks("extract_text", batch_size=5)
    assert claimed == ["/notes/a.md"]
    # Claiming moves the task to PROCESSING, so a second claim finds nothing.
    assert db.claim_tasks("extract_text", batch_size=5) == []

    db.complete_task("/notes/a.md", "extract_text")
    assert db.is_task_done("/notes/a.md", "extract_text")


def test_enqueue_ignores_duplicates(db):
    db.enqueue_task("/notes/a.md", "extract_text")
    db.enqueue_task("/notes/a.md", "extract_text")

    assert db.claim_tasks("extract_text", batch_size=5) == ["/notes/a.md"]


def test_re_enqueue_resets_completed_task(db):
    db.enqueue_task("/notes/a.md", "extract_text")
    db.claim_tasks("extract_text", batch_size=1)
    db.complete_task("/notes/a.md", "extract_text")

    db.re_enqueue_task("/notes/a.md", "extract_text")

    assert not db.is_task_done("/notes/a.md", "extract_text")
    assert db.claim_tasks("extract_text", batch_size=1) == ["/notes/a.md"]


# ── Conversations ────────────────────────────────────────────────────

def test_conversation_message_round_trip(db):
    cid = db.create_conversation(title="Chat")
    db.save_message(cid, "user", "hello")
    db.save_message(cid, "assistant", "hi there")

    messages = db.get_conversation_messages(cid)
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]
    assert db.conversation_message_count(cid) == 2


def test_replace_conversation_messages_packs_tool_calls(db):
    cid = db.create_conversation()
    db.save_message(cid, "user", "stale")

    db.replace_conversation_messages(cid, [
        {"role": "user", "content": "find x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "search"}]},
        {"role": "tool", "content": "result", "tool_call_id": "1", "name": "search"},
    ])

    messages = db.get_conversation_messages(cid)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert "tool_calls" in messages[1]["content"]  # JSON-packed
    assert messages[2]["tool_call_id"] == "1"


def test_delete_conversation_removes_messages(db):
    cid = db.create_conversation()
    db.save_message(cid, "user", "hello")
    db.delete_conversation(cid)

    assert db.get_conversation(cid) is None
    assert db.get_conversation_messages(cid) == []


def test_title_check_threshold_tracks_unseen_messages(db):
    cid = db.create_conversation(title="Untitled")
    for _ in range(5):
        db.save_message(cid, "user", "msg")

    due = db.list_conversations_for_title_check(threshold=4)
    assert [c["id"] for c in due] == [cid]

    # After marking the high-water mark, it's no longer due.
    db.update_conversation_title_check_count(cid, 5)
    assert db.list_conversations_for_title_check(threshold=4) == []


# ── Users ────────────────────────────────────────────────────────────

def test_base_user_is_seeded(db):
    base = db.get_user(DEFAULT_USER_ID)
    assert base is not None
    assert base["config"] == {}
    assert base["user_type"] == "base"
    assert base["username"] is None  # the base user is not a login account


def test_upsert_user_is_idempotent_on_identity(db):
    uid1 = db.upsert_user("art", "guest")
    uid2 = db.upsert_user("art", "guest")
    assert uid1 == uid2
    assert uid1 != DEFAULT_USER_ID


def test_credentials_round_trip_and_username_is_unique(db):
    uid = db.upsert_user("art", "alice@example.com")
    db.set_user_credentials(uid, "alice", "hash123")

    found = db.get_user_by_username("alice")
    assert found is not None and found["id"] == uid
    assert found["password_hash"] == "hash123"

    other = db.upsert_user("art", "bob@example.com")
    with pytest.raises(Exception):  # UNIQUE(username) violation
        db.set_user_credentials(other, "alice", "hash456")


def test_user_config_round_trip(db):
    uid = db.upsert_user("art", "guest")
    db.set_user_config(uid, {"theme": "dark", "credits": 10})
    assert db.get_user_config(uid) == {"theme": "dark", "credits": 10}


def test_user_type_is_frontend_defined_metadata(db):
    uid = db.upsert_user("art", "alice", user_type="creator")
    assert db.get_user(uid)["user_type"] == "creator"

    assert db.upsert_user("art", "alice", user_type="guest") == uid
    assert db.get_user(uid)["user_type"] == "creator"  # touch existing identities; don't reclassify them

    db.set_user_type(uid, "paid")
    assert db.get_user_by_external("art", "alice")["user_type"] == "paid"


def test_conversations_are_user_scoped(db):
    mine = db.create_conversation(title="mine", user_id=DEFAULT_USER_ID)
    theirs = db.create_conversation(title="theirs", user_id=2)

    ids_for_1 = {c["id"] for c in db.list_conversations(user_id=DEFAULT_USER_ID)}
    ids_for_2 = {c["id"] for c in db.list_conversations(user_id=2)}
    assert mine in ids_for_1 and theirs not in ids_for_1
    assert theirs in ids_for_2 and mine not in ids_for_2

    page, _ = db.list_conversations_page(user_id=2)
    assert {c["id"] for c in page} == {theirs}


def test_scoped_delete_is_a_noop_on_mismatch(db):
    cid = db.create_conversation(title="owned by 1", user_id=DEFAULT_USER_ID)
    db.delete_conversation(cid, user_id=2)  # wrong owner → no-op
    assert db.get_conversation(cid) is not None
    db.delete_conversation(cid, user_id=DEFAULT_USER_ID)
    assert db.get_conversation(cid) is None


def test_create_conversation_defaults_to_base_user(db):
    cid = db.create_conversation(title="default owner")
    assert db.get_conversation(cid)["user_id"] == DEFAULT_USER_ID


# ── Direct query ─────────────────────────────────────────────────────

def test_query_rejects_non_select(db):
    with pytest.raises(ValueError):
        db.query("DELETE FROM files")


def test_query_returns_columns_and_rows(db):
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 100.0)
    result = db.query("SELECT path, modality FROM files")

    assert result["columns"] == ["path", "modality"]
    assert result["rows"] == [("/notes/a.md", "text")]
    assert result["truncated"] is False


def test_query_truncates_at_max_rows(db):
    for i in range(5):
        db.upsert_file(f"/notes/{i}.md", f"{i}.md", ".md", "text", float(i))

    result = db.query("SELECT path FROM files", max_rows=2)
    assert len(result["rows"]) == 2
    assert result["truncated"] is True


def test_system_stats_groups_files_and_tasks(db):
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 1.0)
    db.enqueue_task("/notes/a.md", "extract_text")

    stats = db.get_system_stats()
    assert stats["files"]["text"] == 1
    assert stats["tasks"]["extract_text"]["PENDING"] == 1
