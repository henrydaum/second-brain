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
from sandbox.handlers.kernel import _db_define, _db_write
from sandbox.policy import SAFE, UNSAFE, Chain, classify
from sandbox.users import KERNEL_TABLES, ScopeError, scope_write
from sandbox.validator import validate_file


def _chain():
    return Chain(root="user").push("some_plugin")


# ── db.write may not reach what db.query is scoped away from ──────────

@pytest.mark.parametrize("sql", [
    "UPDATE users SET password_hash='x' WHERE id=1",
    "DELETE FROM conversations",
    "UPDATE conversations SET user_id=2",
    "INSERT INTO action_ledger (origin) VALUES ('forged')",
    "DELETE FROM conversation_messages",
])
def test_kernel_tables_are_not_writable_through_a_request(sql):
    """``db.write`` was the way around the rest of the catalogue.

    ``conv.delete`` prompts and ``DELETE FROM conversations`` did not;
    ``user.write`` prompts and ``UPDATE users SET password_hash`` did not.
    Both are the same act, so both get the same answer.
    """
    with pytest.raises(ScopeError):
        scope_write(sql)

    assert _db_write(None, {"sql": sql}).denied


def test_ddl_cannot_redefine_a_kernel_table():
    """Dropping the table is the obvious next try once writing is refused."""
    assert _db_define(None, {"ddl": "DROP TABLE action_ledger"}).denied
    assert _db_write(None, {"sql": "ALTER TABLE users ADD COLUMN x"}).denied


@pytest.mark.parametrize("sql", [
    "INSERT INTO plugin_notes (body) VALUES ('hi')",
    "CREATE TABLE my_notes_files (id INTEGER)",
    "UPDATE search_index SET score = 1",
])
def test_plugin_owned_tables_stay_freely_writable(sql):
    """The check is a table list, not a ban on writing.

    ``my_notes_files`` is the interesting one: it contains ``files``, and a
    substring match rather than a word match would refuse a plugin's own
    table and make ``db.define`` useless for anything with a common noun in
    its name.
    """
    assert scope_write(sql) == sql


def test_a_refused_write_is_a_denial_not_a_breakage():
    """``except sdk.Denied`` has to catch policy, or the split means nothing."""
    assert _db_write(None, {"sql": "DELETE FROM users"}).denied


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


def test_an_approval_gated_command_is_refused_rather_than_dispatched():
    """The state machine owns that answer; nothing here can supply it.

    Dispatching with ``_approved=True`` from this path would forge exactly the
    consent the approval mechanism exists to obtain.
    """
    from sandbox.handlers.kernel import _command_call

    class Gated:
        require_approval = True

    class Registry:
        _commands = {"update": Gated()}

        def dispatch_dict(self, *a, **k):        # pragma: no cover - refused
            raise AssertionError("must not dispatch an approval-gated command")

    class Ctx:
        command_registry = Registry()
        session_key = "s"

    assert _command_call(Ctx(), {"name": "update"}).denied


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
