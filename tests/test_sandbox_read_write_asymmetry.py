"""The write paths must not go around the controls the read paths carry.

Every hole fixed here had the same shape: a narrow, carefully-scoped Request
sat next to a broad one that reached the same thing without the scoping. A
control the catalogue enforces in one place and not in the place next to it is
not a control, so each of these is pinned rather than described.
"""

from pathlib import Path

import pytest

from sandbox.guest.requests import (COMMAND_CALL, DB_QUERY, DB_WRITE, FS_READ,
                                    FS_READ_BYTES, TOOL_CALL, Request)
from sandbox.handlers.fs_net import _fs_read, _fs_read_bytes, _fs_search
from sandbox.handlers.kernel import (_db_define, _db_write,
                                     _session_add_attachment)
from sandbox.policy import SAFE, UNSAFE, Chain, classify
from sandbox.users import KERNEL_TABLES, ScopeError, scope_write
from sandbox.validator import validate_file


def _chain():
    return Chain(root="user").push("some_plugin")


# ── db.write: kernel rows are data, kernel schemas are structure ──────
#
# The line here moved once, and the reason is worth keeping. It used to be
# "kernel tables are not writable at all", argued from ``db.write`` being a way
# around ``conv.delete``'s dialog. What that actually blocked was bookkeeping:
# not every kernel *column* has a Request behind it, so a task maintaining one
# had to mirror rows it could already read into a shadow table. Meanwhile the
# integrity fear was unfounded — ``ON DELETE CASCADE`` plus
# ``PRAGMA foreign_keys = ON`` means raw SQL cascades exactly as ``conv.delete``
# does, that being a bare DELETE itself.
#
# So: data is writable, structure is not, and the one irreversible data write
# is asked about rather than refused.

@pytest.mark.parametrize("sql", [
    "UPDATE conversations SET last_title_check_message_count = 4 WHERE id = 1",
    "INSERT INTO action_ledger (origin) VALUES ('plugin')",
    "UPDATE users SET user_type = 'paid' WHERE id = 2",
    "UPDATE conversation_messages SET content = 'edited' WHERE id = 9",
])
def test_kernel_table_rows_are_writable(sql):
    """Changing data cannot change how the kernel works; only structure can."""
    assert scope_write(sql) == sql
    assert classify(Request(DB_WRITE, {"sql": sql}), _chain()).level == SAFE


@pytest.mark.parametrize("sql", [
    "DROP TABLE action_ledger",
    "ALTER TABLE users ADD COLUMN x",
    "ALTER TABLE conversations DROP COLUMN title",
    "CREATE TABLE conversations (id INTEGER)",
])
def test_kernel_table_schemas_are_not(sql):
    """A dropped column breaks every query the kernel makes against it."""
    with pytest.raises(ScopeError):
        scope_write(sql)
    assert _db_write(None, {"sql": sql}).denied


def test_ddl_is_refused_through_db_define_too():
    """``db.define`` is the other door onto the same act."""
    assert _db_define(None, {"ddl": "DROP TABLE action_ledger"}).denied


@pytest.mark.parametrize("sql", [
    "PRAGMA writable_schema = ON",
    "UPDATE sqlite_master SET sql = 'x'",
    "UPDATE sqlite_schema SET sql = 'x'",
])
def test_ddl_wearing_dml_s_clothes_is_still_ddl(sql):
    """The bypass that makes the keyword check honest.

    DDL is not only spelled CREATE: ``PRAGMA writable_schema=ON`` followed by
    an ``UPDATE sqlite_master`` is schema surgery that starts with UPDATE.
    Without these two refusals the rule above would ship with a hole in it.
    """
    with pytest.raises(ScopeError):
        scope_write(sql)


@pytest.mark.parametrize("sql", [
    "ATTACH DATABASE '/etc/passwd' AS other",
    "DETACH other",
    "VACUUM",
])
def test_statements_that_act_on_the_file_are_refused(sql):
    """These name no table, so a per-table check waves them straight through.

    ``ATTACH`` is the sharp one: it opens an arbitrary path and exposes it to
    SQL, which is filesystem access spelled in a language the filesystem rules
    do not read.
    """
    with pytest.raises(ScopeError):
        scope_write(sql)


def test_a_credential_column_is_never_writable():
    """Unwritable for the same reason it is unreadable, and not a table rule.

    Every other column on ``users`` is ordinary metadata a frontend may
    maintain; this one has no bookkeeping use and an unrecoverable accident.
    """
    with pytest.raises(ScopeError):
        scope_write("UPDATE users SET password_hash='x' WHERE id=1")
    assert _db_write(None, {"sql": "UPDATE users SET password_hash='x'"}).denied


@pytest.mark.parametrize("sql", [
    "DELETE FROM conversations WHERE id = 5",
    "DELETE FROM conversation_messages",
    "DELETE FROM users WHERE id = 2",
])
def test_deleting_kernel_rows_is_asked_about(sql):
    """Legitimate, and irreversible — which is what a dialog is for.

    An UPDATE with a bad WHERE clause leaves the rows there to fix. A DELETE
    with the same bad WHERE clause leaves somebody asking where their
    conversations went, so this one is asked rather than assumed.
    """
    assert scope_write(sql) == sql
    decision = classify(Request(DB_WRITE, {"sql": sql}), _chain())
    assert decision.level == UNSAFE
    assert "delete" in decision.reason


def test_a_leading_comment_does_not_hide_the_verb():
    """The first *word* is the check, and a statement may open with a comment."""
    sql = "-- tidy up\nDELETE FROM conversations"
    assert classify(Request(DB_WRITE, {"sql": sql}), _chain()).level == UNSAFE


@pytest.mark.parametrize("sql", [
    "INSERT INTO plugin_notes (body) VALUES ('hi')",
    "CREATE TABLE my_notes_files (id INTEGER)",
    "DROP TABLE my_notes_files",
    "UPDATE search_index SET score = 1",
    "DELETE FROM search_index WHERE stale = 1",
])
def test_plugin_owned_tables_stay_freely_writable(sql):
    """The check is a table list, not a ban on writing.

    ``my_notes_files`` is the interesting one: it contains ``files``, and a
    substring match rather than a word match would refuse a plugin's own
    table and make ``db.define`` useless for anything with a common noun in
    its name. The DELETE matters too — a dialog every time a plugin tidies its
    own cache is how people learn to stop reading dialogs.
    """
    assert scope_write(sql) == sql
    assert classify(Request(DB_WRITE, {"sql": sql}), _chain()).level == SAFE


def test_a_refused_write_is_a_denial_not_a_breakage():
    """``except sdk.Denied`` has to catch policy, or the split means nothing."""
    assert _db_write(None, {"sql": "DROP TABLE users"}).denied


def test_every_kernel_table_names_the_request_that_replaces_it():
    """A refusal that does not say what to do instead only teaches evasion."""
    assert all(v.startswith("sdk.") for v in KERNEL_TABLES.values())


# ── fs.read may not reach what secret.reveal is gated on ──────────────

def _config_path():
    from config.config_manager import _DEFAULT_CONFIG_PATH
    return _DEFAULT_CONFIG_PATH


def test_the_config_file_is_not_readable_as_a_file():
    """Otherwise the secret handle mechanism is decorative.

    ``config.read`` returns ``<secret:name>`` and ``secret.reveal`` prompts,
    but ``config.json`` holds the same values in plaintext and ``fs.read`` is
    safe for any path — so the front door was locked and the file sat open
    beside it.
    """
    result = _fs_read(None, {"path": _config_path()})
    assert result.denied
    assert "secret_" in result.error


def test_the_encoding_asked_for_is_not_a_way_around_it():
    """``fs.read_bytes`` returns the same bytes with a different wrapper."""
    assert _fs_read_bytes(None, {"path": _config_path()}).denied


def test_search_cannot_grep_a_protected_file():
    """``fs.search`` returns matching *lines*, so it leaks as readily.

    Hits are skipped rather than the whole search refused: a scan over a tree
    should not fail because a protected file happens to sit inside it.
    """
    root = str(Path(_config_path()).parent)
    hits = _fs_search(None, {"pattern": "secret_", "root": root,
                             "glob": "*.json"})
    assert hits.ok
    assert hits.data == []


def test_reads_elsewhere_are_untouched(tmp_path):
    """The deny-list is a few named files, not a new sandbox on the disk."""
    ordinary = tmp_path / "notes.txt"
    ordinary.write_text("hello", encoding="utf-8")
    assert _fs_read(None, {"path": str(ordinary)}).data == "hello"


def test_staging_an_attachment_cannot_reach_a_protected_file():
    """``session.add_attachment`` is a read, and reads the *kernel* performs.

    The guest names a path and the kernel opens it and puts the contents in
    front of the model — so a route that skipped the deny-list would read
    ``config.json`` aloud to a provider, which is precisely the leak the two
    tests above exist to close. This is the third door onto the same bytes.
    """
    result = _session_add_attachment(None, {"path": _config_path()})
    assert result.denied
    assert "secret_" in result.error


# ── command.call carries the user's authority, so it is asked about ───

def test_calling_a_slash_command_is_unsafe():
    """A command is what the *person* types, and the set includes /packages.

    Unlike a tool, it is not narrowed by the agent's scope and is not written
    to be called by other code, so running one on somebody's behalf is worth
    a sentence. The dialog names it.
    """
    decision = classify(Request(COMMAND_CALL, {"name": "packages"}), _chain())
    assert decision.level == UNSAFE
    assert "/packages" in decision.reason


def test_calling_a_tool_stays_safe():
    """Provenance is what makes this fine, and it is worth pinning.

    The callee's own Requests are classified with the caller still in the
    chain, so routing through a tool launders nothing.
    """
    assert classify(Request(TOOL_CALL, {"name": "x"}), _chain()).level == SAFE


def test_an_approval_gated_command_receives_the_command_call_approval():
    """Only a policy-approved command.call reaches the handler."""
    from sandbox.handlers.kernel import _command_call

    class Gated:
        require_approval = True

        def requires_approval(self, args):
            return args.get("action") == "update"

    class Registry:
        _commands = {"update": Gated()}
        calls = []

        def dispatch_dict(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "done"

    class Ctx:
        command_registry = Registry()
        session_key = "s"

    result = _command_call(
        Ctx(), {"name": "update", "args": {"action": "update"}})
    assert result.ok and result.data == "done"
    assert Ctx.command_registry.calls[0][1]["_approved"] is True


def test_an_ordinary_command_call_does_not_gain_a_nested_request_grant():
    """Approval to use the command surface grants only declared gated actions."""
    from sandbox.handlers.kernel import _command_call

    class Ordinary:
        require_approval = False

        def requires_approval(self, _args):
            return False

    class Registry:
        _commands = {"show": Ordinary()}
        approved = None

        def dispatch_dict(self, _name, _args, **kwargs):
            self.approved = kwargs["_approved"]
            return "shown"

    class Ctx:
        command_registry = Registry()
        session_key = "s"

    assert _command_call(Ctx(), {"name": "show"}).data == "shown"
    assert Ctx.command_registry.approved is False


# ── the linter closes the escape sitting next to a rule it enforces ───

def test_the_builtins_namespace_is_not_reachable(tmp_path):
    """Banning ``open`` while leaving ``__builtins__`` open stops nobody.

    The linter is not a proof and is not meant to be — but an escape directly
    beside an enforced rule is worth the one line.
    """
    source = tmp_path / "tool_sneaky.py"
    source.write_text(
        "from guest.bases import BaseTool\n"
        "requests = []\n\n"
        "class SneakyTool(BaseTool):\n"
        "    name = 'sneaky'\n"
        "    description = 'x'\n\n"
        "    def run(self, sdk):\n"
        "        return getattr(__builtins__, 'open')('x')\n",
        encoding="utf-8")

    report = validate_file(source)
    assert not report.ok
    assert "__builtins__" in report.render()


# ── secret.reveal is gated by ownership, not by need ──────────────────
#
# CLAUDE.md described this rule for a long time before the code carried it:
# a plugin reading back the credential it declared is not asked, because
# configuring the setting was the consent. Everyone else is asked. The gap
# had teeth — a frontend needs its own token inside start(), before any
# frontend exists to draw a dialog, so an unconditional ask there is a
# question nobody can answer, at boot, every boot.

def _reveal(name: str):
    from sandbox.guest.requests import SECRET_REVEAL

    return Request(SECRET_REVEAL, {"name": name})


def _owned_by(monkeypatch, key: str, owners: list):
    """Make the setting registry report ``owners`` as declaring ``key``."""
    import plugins.plugin_discovery as discovery

    monkeypatch.setattr(
        discovery, "get_setting_plugin_names",
        lambda setting: list(owners) if setting == key else [])


def test_a_plugin_may_read_back_the_secret_it_declared(monkeypatch):
    _owned_by(monkeypatch, "secret_telegram_bot_token", ["telegram"])

    chain = Chain(root="frontend:telegram").push("frontend_telegram")
    decision = classify(_reveal("secret_telegram_bot_token"), chain)

    assert decision.level == SAFE
    assert "telegram" in decision.reason


def test_someone_elses_secret_is_still_asked_about(monkeypatch):
    """The exemption is ownership. Needing the value is not a claim to it."""
    _owned_by(monkeypatch, "secret_telegram_bot_token", ["telegram"])

    chain = Chain(root="user").push("tool_exfiltrate")
    decision = classify(_reveal("secret_telegram_bot_token"), chain)

    assert decision.level == UNSAFE


def test_an_undeclared_secret_is_asked_about(monkeypatch):
    """No owner means nobody can claim it — an env var, say."""
    _owned_by(monkeypatch, "secret_telegram_bot_token", ["telegram"])

    chain = Chain(root="frontend:telegram").push("frontend_telegram")

    assert classify(_reveal("OPENAI_API_KEY"), chain).level == UNSAFE
    assert classify(_reveal(""), chain).level == UNSAFE


def test_ownership_cannot_be_claimed_by_the_caller(monkeypatch):
    """The identity comes from the kernel-assigned root and the chain links,
    never from anything the guest says about itself."""
    _owned_by(monkeypatch, "secret_telegram_bot_token", ["telegram"])

    # A plugin that merely names itself convincingly in its own arguments.
    chain = Chain(root="user").push("impostor")
    request = Request("secret.reveal", {"name": "secret_telegram_bot_token",
                                        "owner": "telegram",
                                        "plugin": "telegram"})

    assert classify(request, chain).level == UNSAFE


def test_a_callee_does_not_inherit_the_owner_s_exemption_by_name(monkeypatch):
    """Ownership is set membership over the chain, so a tool called *by* the
    owner is covered — that is the caller spending its own grant — while an
    unrelated chain is not."""
    _owned_by(monkeypatch, "secret_telegram_bot_token", ["telegram"])

    inside = Chain(root="frontend:telegram").push("frontend_telegram").push(
        "tool_helper")
    outside = Chain(root="frontend:mcp_server").push("frontend_mcp_server")

    assert classify(_reveal("secret_telegram_bot_token"), inside).level == SAFE
    assert classify(_reveal("secret_telegram_bot_token"),
                    outside).level == UNSAFE
