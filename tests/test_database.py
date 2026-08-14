"""Tests for the SQLite layer (``pipeline.database.Database``).

The database backs two core kernel concerns: the file/task pipeline queue and
durable conversation storage. These tests run against a fresh on-disk DB in a
temp dir, so the schema bootstrap in ``_setup`` is exercised for real.
"""

import sqlite3

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


def test_conversation_page_walks_with_offset(db):
    # Newest first, so the ids come back in reverse creation order.
    ids = [db.create_conversation(title=f"c{n}", user_id=DEFAULT_USER_ID)
           for n in range(5)]

    first, more = db.list_conversations_page(offset=0, limit=2,
                                             user_id=DEFAULT_USER_ID)
    assert [c["id"] for c in first] == ids[:-3:-1]
    assert more is True

    second, more = db.list_conversations_page(offset=2, limit=2,
                                              user_id=DEFAULT_USER_ID)
    assert [c["id"] for c in second] == ids[2:0:-1]
    assert more is True

    last, more = db.list_conversations_page(offset=4, limit=2,
                                            user_id=DEFAULT_USER_ID)
    assert [c["id"] for c in last] == [ids[0]]
    assert more is False

    # Past the end is an empty page, not an error.
    beyond, more = db.list_conversations_page(offset=99, limit=2,
                                              user_id=DEFAULT_USER_ID)
    assert beyond == [] and more is False


def test_category_counts_span_the_table_not_a_page(db):
    for _ in range(3):
        db.create_conversation(title="sub", category="Subagent",
                               user_id=DEFAULT_USER_ID)
    db.create_conversation(title="plain", user_id=DEFAULT_USER_ID)
    # NULL and empty are one bucket, the way ``category=""`` filters them.
    db.create_conversation(title="blank", category="", user_id=DEFAULT_USER_ID)

    counts = dict(db.count_conversations_by_category(user_id=DEFAULT_USER_ID))
    assert counts[None] == 2
    assert counts["Subagent"] == 3

    # Main first, so a picker built from this lists it where people expect.
    assert db.count_conversations_by_category(
        user_id=DEFAULT_USER_ID)[0][0] is None

    # Another owner's rows are not counted into yours.
    db.create_conversation(title="theirs", category="Subagent", user_id=2)
    assert dict(db.count_conversations_by_category(
        user_id=DEFAULT_USER_ID))["Subagent"] == 3


def test_conv_list_pages_and_reports_whole_table_counts(db):
    """The handler's own contract, not just the SQL underneath it.

    ``offset`` was hardcoded to 0 here long after ``list_conversations_page``
    grew the argument, so a client could ask for page two and be handed page
    one - which looks like a list that simply stops.
    """
    from types import SimpleNamespace

    from sandbox.guest.requests import CONV_LIST
    from tests.support import call_handler

    # The categorised one first, so it is the *oldest* and the four below stay
    # the front of the unfiltered list.
    db.create_conversation(title="sub", category="Subagent",
                           user_id=DEFAULT_USER_ID)
    ids = [db.create_conversation(title=f"c{n}", user_id=DEFAULT_USER_ID)
           for n in range(4)]
    ctx = SimpleNamespace(db=db, user_id=DEFAULT_USER_ID)

    first = call_handler(CONV_LIST, ctx, {"details": True, "limit": 2})
    assert [c["id"] for c in first.data["items"]] == [ids[3], ids[2]]
    assert first.data["has_more"] is True

    second = call_handler(
        CONV_LIST, ctx, {"details": True, "limit": 2, "offset": 2})
    assert [c["id"] for c in second.data["items"]] == [ids[1], ids[0]]

    # Counts describe the table, so the same numbers come back on every page.
    for answer in (first, second):
        counts = {e["category"]: e["count"] for e in answer.data["categories"]}
        assert counts[None] == 4 and counts["Subagent"] == 1

    # And the Main bucket is reachable on its own.
    main = call_handler(CONV_LIST, ctx, {"details": True, "category": ""})
    assert {c["id"] for c in main.data["items"]} == set(ids)


def test_main_bucket_is_asked_for_with_an_empty_category(db):
    plain = db.create_conversation(title="plain", user_id=DEFAULT_USER_ID)
    blank = db.create_conversation(title="blank", category="",
                                   user_id=DEFAULT_USER_ID)
    db.create_conversation(title="sub", category="Subagent",
                           user_id=DEFAULT_USER_ID)

    # `""` is the Main bucket; `None` is no filter at all. That distinction is
    # what lets a client ask for uncategorised conversations without reading
    # every row to find them.
    main, _ = db.list_conversations_page(category="", user_id=DEFAULT_USER_ID)
    assert {c["id"] for c in main} == {plain, blank}

    every, _ = db.list_conversations_page(category=None,
                                          user_id=DEFAULT_USER_ID)
    assert len(every) == 3


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


def test_read_queries_allow_only_schema_introspection_pragmas(db):
    result = db.query("PRAGMA table_info(files)")
    assert "name" in result["columns"]

    rows = db.query_rows("PRAGMA main.table_info(files)")
    assert any(row["name"] == "path" for row in rows)


def test_a_read_can_match_against_an_fts5_index(db):
    """A full-text search is a read, and the authorizer used to think otherwise.

    FTS5 asks SQLite for ``PRAGMA data_version`` from inside the virtual table
    — nothing in the query mentions a PRAGMA — and the read authorizer denied
    every PRAGMA but ``table_info``. So *every* ``MATCH`` failed with a bare
    "authorization denied", which meant the store's ``lexical_search`` could
    not run a single search from a sandboxed tool. It reported that as "no
    results", so the keyword half of retrieval looked like an empty corpus
    rather than a refusal, and stayed that way for a long time.

    Written against a real FTS5 table rather than by asserting the pragma is
    allowed: the bug was that a query nobody would think to check was refused,
    so the test has to be the query.
    """
    db.execute_write("CREATE VIRTUAL TABLE notes_fts USING fts5(path, content)")
    db.execute_write("INSERT INTO notes_fts (path, content)"
                     " VALUES ('/notes/a.md', 'retry a failed upload')")
    db.execute_write("INSERT INTO notes_fts (path, content)"
                     " VALUES ('/notes/b.md', 'something else entirely')")

    rows = db.query_rows(
        "SELECT path FROM notes_fts WHERE notes_fts MATCH ? ORDER BY rank",
        ["upload"])

    assert [row["path"] for row in rows] == ["/notes/a.md"]


def test_a_refused_read_says_what_it_tried_to_do(db):
    """SQLite's own message for this is the two words "not authorized".

    No statement, no construct, no hint — and the construct is often not in
    the query at all. That is a rule nobody can act on, which is how a working
    guard gets read as a broken database and retried verbatim.

    The CTE is the case that proves the guard is load-bearing rather than
    decorative: it opens with WITH, so the statement-level prefix check waves
    it through, and only the authorizer stops it being a write.
    """
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 100.0)

    with pytest.raises(sqlite3.DatabaseError) as caught:
        db.query_rows("WITH c AS (SELECT path FROM files) DELETE FROM files")

    assert "tried to delete files" in str(caught.value)
    assert "db.write" in str(caught.value)
    assert db.get_all_files() == {"/notes/a.md": 100.0}


def test_refusals_name_the_construct_and_not_a_lookalike():
    """``sqlite3`` overloads the action integers three ways.

    The authorizer's return values (SQLITE_DENY is 1) and the SQLITE_LIMIT_*
    constants share the space with the actions — SQLITE_DELETE and
    SQLITE_LIMIT_VARIABLE_NUMBER are both 9. Sweeping the namespace therefore
    renders a refused DELETE as "limit variable number": a message written to
    stop people guessing, confidently naming the wrong construct.
    """
    from pipeline.database import _describe_action

    assert _describe_action(sqlite3.SQLITE_DELETE, "files", None) == "delete files"
    assert _describe_action(sqlite3.SQLITE_PRAGMA, "table_list", None) == (
        "pragma table_list")
    # Whatever the collisions are named, nothing may resolve to one of them.
    limits = {getattr(sqlite3, name) for name in dir(sqlite3)
              if name.startswith("SQLITE_LIMIT_")}
    for code in limits:
        assert "limit" not in _describe_action(code, "", None)


def test_data_version_stays_readable_but_unsettable(db):
    """The pragma is allowed because it is an *answer*, not a switch.

    It reports whether another connection has committed and takes no argument,
    so there is no form of it that changes anything. The guard still refuses
    the statement-level route, because a plugin has no reason to ask and the
    allowance exists for SQLite's own internals.
    """
    with pytest.raises(ValueError, match="PRAGMA table_info"):
        db.query_rows("PRAGMA data_version")


@pytest.mark.parametrize("sql", [
    "PRAGMA foreign_keys = OFF",
    "PRAGMA writable_schema = ON",
    "PRAGMA wal_checkpoint(TRUNCATE)",
    "PRAGMA journal_mode = DELETE",
    "PRAGMA optimize",
])
def test_read_queries_reject_stateful_pragmas(db, sql):
    with pytest.raises(ValueError, match="PRAGMA table_info"):
        db.query_rows(sql)

    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_a_cte_cannot_disguise_a_write_as_a_read(db):
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 100.0)

    with pytest.raises(sqlite3.DatabaseError):
        db.query_rows(
            "WITH chosen AS (SELECT path FROM files) "
            "DELETE FROM files WHERE path IN (SELECT path FROM chosen)"
        )

    assert db.get_all_files() == {"/notes/a.md": 100.0}


def test_system_stats_groups_files_and_tasks(db):
    db.upsert_file("/notes/a.md", "a.md", ".md", "text", 1.0)
    db.enqueue_task("/notes/a.md", "extract_text")

    stats = db.get_system_stats()
    assert stats["files"]["text"] == 1
    assert stats["tasks"]["extract_text"]["PENDING"] == 1
