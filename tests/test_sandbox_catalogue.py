"""The Request catalogue wired to handlers: completeness, secrets, scoping.

Three properties matter more than any individual handler:

- every Request in the catalogue is *classified* — nothing reaches an effect
  without a decision
- every Request is either serviced or explicitly listed as unwired
- the SDK can actually reach everything the catalogue defines
"""

import sqlite3
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Chain, Interpreter, Request, Result
from sandbox.guest import requests as R
from sandbox.guest.sdk import SDK
from sandbox.handlers import HANDLERS, UNWIRED
from sandbox.policy import classify
from sandbox.credentials import handle_for, is_secret, redact, resolve
from sandbox.users import ScopeError, scope_sql
from tests.support import call_handler


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Completeness.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@pytest.mark.parametrize("kind", sorted(R.ALL_TYPES))
def test_every_request_is_classified(kind):
    """Nothing reaches an effect without a decision being made about it."""
    decision = classify(Request(kind, {}), Chain())
    assert decision.level in ("safe", "unsafe")
    assert "unclassified" not in decision.reason


@pytest.mark.parametrize("kind", sorted(R.ALL_TYPES))
def test_every_request_is_serviced_or_declared_unwired(kind):
    """'Not built yet' must be distinguishable from 'misspelled'."""
    assert kind in HANDLERS or kind in UNWIRED or kind == R.SELF_RESPOND


def test_nothing_in_the_catalogue_is_unwired():
    """Every capability the vocabulary names is now actually serviced.

    This was an allowlist of deliberate holes, shrinking as each was closed.
    ``agent.spawn`` and ``agent.schedule`` were the last two, and they closed
    when subagents moved into the kernel. An empty inventory is a stronger
    statement than a shrinking one: a Request added without a handler now
    fails here rather than being quietly added to a list.
    """
    assert not UNWIRED


def test_plugin_lifecycle_handlers_share_the_kernel_watcher():
    """SDK mutations enter the same coordinator as filesystem events."""
    source = Path("bundled/commands/command_clear.py").resolve()

    class Watcher:
        def __init__(self):
            self.calls = []

        def register(self, path):
            self.calls.append(("register", Path(path)))
            return {"ok": True, "name": "clear", "family": "command",
                    "path": str(path)}

        def unregister(self, path):
            self.calls.append(("unregister", Path(path)))
            return {"ok": True, "names": ["clear"], "family": "command",
                    "path": str(path)}

        def reload(self, path):
            self.calls.append(("reload", Path(path)))
            return {"ok": True, "name": "clear", "family": "command",
                    "path": str(path)}

        def resolve_registered(self, name, family):
            assert (name, family) == ("clear", "command")
            return source, None

    watcher = Watcher()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(plugin_watcher=watcher),
    )
    registered = call_handler(R.PLUGIN_REGISTER, ctx, {"path": str(source)})
    unloaded = call_handler(
        R.PLUGIN_UNREGISTER, ctx, {"name": "clear", "family": "command"})
    reloaded = call_handler(R.PLUGIN_RELOAD, ctx, {"path": str(source)})

    assert registered.ok and unloaded.ok and reloaded.ok
    assert watcher.calls == [
        ("register", source),
        ("unregister", source),
        ("reload", source),
    ]


def test_plugin_lifecycle_rejects_paths_outside_plugin_roots(tmp_path):
    """Approval never turns an arbitrary filesystem path into a plugin."""
    class Watcher:
        def register(self, _path):
            raise AssertionError("outside path reached coordinator")

    ctx = SimpleNamespace(
        runtime=SimpleNamespace(plugin_watcher=Watcher()),
    )
    result = call_handler(
        R.PLUGIN_REGISTER, ctx, {"path": str(tmp_path / "tool_bad.py")})

    assert not result.ok
    # Pin that the refusal is *about the location* by naming the rejected
    # directory back, rather than pinning the sentence that says so.
    assert str(tmp_path) in result.error


def test_the_sdk_reaches_every_wired_request():
    """A catalogue nothing can call is a catalogue that does not exist.

    Every SDK method is invoked with placeholder arguments derived from its
    own signature, and the Requests it emits are collected. Anything wired to
    a handler but unreachable from the SDK is a hole.
    """
    import inspect

    sent = []

    class Recorder:
        """Captures Requests instead of servicing them."""

        def send(self, request):
            """Record and answer blandly."""
            sent.append(request.type)
            return Result(data=None)

        def log(self, level, message):
            """Ignore."""

    from sandbox.guest.sdk import _Namespace

    sdk = SDK(Recorder())
    for name in [n for n in vars(sdk) if not n.startswith("_")]:
        namespace = getattr(sdk, name)
        # Pure helpers make no Requests; only the Request namespaces matter.
        if not isinstance(namespace, _Namespace):
            continue
        for attr in dir(namespace):
            if attr.startswith("_"):
                continue
            method = getattr(namespace, attr)
            if not inspect.isfunction(method) and not inspect.ismethod(method):
                continue
            signature = inspect.signature(method)
            args = {}
            for parameter in signature.parameters.values():
                if parameter.default is not inspect.Parameter.empty:
                    continue
                if parameter.kind in (parameter.VAR_POSITIONAL,
                                      parameter.VAR_KEYWORD):
                    continue
                args[parameter.name] = "x"
            method(**args)

    unreachable = set(HANDLERS) - set(sent)
    assert unreachable == set(), f"no SDK route to: {sorted(unreachable)}"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Policy, family by family.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_egress_is_unsafe_whatever_the_verb():
    """The single control that makes generous reads safe."""
    for method in ("GET", "POST", "HEAD"):
        decision = classify(
            Request(R.NET_HTTP, {"url": "https://x.test/?d=1",
                                 "method": method}), Chain())
        assert not decision.safe


def test_writing_to_scratch_is_safe_and_elsewhere_is_not(tmp_path):
    """Level is a property of the arguments, never of the Request type."""
    import tempfile
    scratch = f"{tempfile.gettempdir()}/sb-test-note.txt"
    assert classify(Request(R.FS_WRITE, {"path": scratch, "data": "x"}),
                    Chain()).safe
    assert not classify(Request(R.FS_WRITE, {"path": "main.pyw", "data": "x"}),
                        Chain()).safe


def test_deleting_outside_scratch_is_unsafe():
    """Destructive by default, wherever it points."""
    assert not classify(Request(R.FS_DELETE, {"path": "main.pyw"}),
                        Chain()).safe


def test_widening_is_unsafe_and_narrowing_is_safe():
    """The rule that runs through the whole catalogue."""
    pairs = [
        (R.SESSION_ADD_TOOL, R.SESSION_REMOVE_TOOL),
        (R.SESSION_ADD_PROMPT, R.SESSION_REMOVE_PROMPT),
    ]
    for widen, narrow in pairs:
        assert not classify(Request(widen, {"tool": "t", "text": "t"}),
                            Chain()).safe, widen
        assert classify(Request(narrow, {"tool": "t", "handle": 1}),
                        Chain()).safe, narrow


def test_self_extension_is_always_unsafe():
    """The literal subject of the LibOS quote."""
    for kind in (R.PLUGIN_REGISTER, R.PLUGIN_INSTALL, R.PLUGIN_UNINSTALL,
                 R.SERVICE_LOAD, R.CONFIG_WRITE):
        assert not classify(Request(kind, {}), Chain()).safe, kind


def test_unattended_work_creation_is_unsafe():
    """Nobody will be present to answer for what it does later."""
    for kind in (R.CRON_CREATE, R.AGENT_SCHEDULE):
        assert not classify(Request(kind, {}), Chain()).safe, kind
    assert classify(Request(R.CRON_LIST, {}), Chain()).safe


def test_asking_a_human_needs_a_human():
    """ui.ask is the approval channel, so it is safe only when attended."""
    attended = Chain(root="user")
    unattended = Chain(root="cron:nightly")
    assert classify(Request(R.UI_ASK, {"prompt": "?"}), attended).safe
    assert not classify(Request(R.UI_ASK, {"prompt": "?"}), unattended).safe


def test_reads_stay_broad():
    """Free reads are safe because the exits are gated, not because they
    are harmless."""
    for kind in (R.FS_READ, R.FS_LIST, R.DB_QUERY, R.CONV_READ, R.LEDGER_READ):
        assert classify(Request(kind, {"path": "x", "sql": "select 1"}),
                        Chain()).safe, kind


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Secret handles.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_the_lint_heuristic_recognises_credential_names():
    """Not policy any more - this is what the validator warns about, and
    what environment variables are still judged by."""
    from sandbox.credentials import looks_secret

    for name in ("brave_api_key", "OPENAI_API_KEY", "client_secret",
                 "db_password", "access_token"):
        assert looks_secret(name), name
    for name in ("model", "max_tokens", "db_path"):
        assert not looks_secret(name), name


def test_a_secret_reads_back_as_a_handle():
    """The sandbox cannot leak what it was never given."""
    assert (redact("secret_brave_api_key", "sk-real")
            == "<secret:secret_brave_api_key>")
    assert redact("model", "opus") == "opus"


def test_handles_resolve_on_the_way_out():
    """Substitution happens in the handler, after policy has decided."""
    lookup = {"secret_brave_api_key": "sk-real"}.get
    payload = {"headers": {"X-Key": handle_for("secret_brave_api_key")},
               "list": [handle_for("secret_brave_api_key")]}
    resolved = resolve(payload, lookup)
    assert resolved["headers"]["X-Key"] == "sk-real"
    assert resolved["list"] == ["sk-real"]


def test_an_unknown_handle_is_left_visible():
    """A literal <secret:foo> reaching a server is a bug you can see; an
    empty string silently substituted is one you cannot."""
    resolved = resolve(handle_for("nope"), {}.get)
    assert resolved == "<secret:nope>"


def test_config_read_redacts(tmp_path):
    """The clause belongs on config, not on the database."""
    ctx = type("Ctx", (), {"config": {"secret_brave_api_key": "sk-real",
                                      "model": "opus"}})()
    handler = partial(call_handler, R.CONFIG_READ)
    assert (handler(ctx, {"key": "secret_brave_api_key"}).data
            == "<secret:secret_brave_api_key>")
    assert handler(ctx, {"key": "model"}).data == "opus"
    everything = handler(ctx, {"key": None}).data
    assert everything["secret_brave_api_key"] == "<secret:secret_brave_api_key>"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Per-user scoping.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_the_virtual_table_expands_to_the_current_user():
    """Scoping is structural: the plugin cannot forget to filter."""
    sql, _ = scope_sql("SELECT * FROM my_conversations", [], 7)
    assert "user_id = 7" in sql
    assert "FROM (SELECT * FROM conversations" in sql


def test_reading_the_base_table_is_refused_with_a_pointer():
    """The same teaching-error approach the validator takes."""
    with pytest.raises(ScopeError) as exc:
        scope_sql("SELECT * FROM conversations", [], 7)
    assert "my_conversations" in str(exc.value)


def test_password_hash_is_never_readable():
    """The only genuinely secret column in the schema."""
    with pytest.raises(ScopeError):
        scope_sql("SELECT password_hash FROM users", [], 1)


def test_unscoped_tables_pass_through_untouched():
    """Reads stay broad; only identity is narrowed."""
    sql, params = scope_sql("SELECT * FROM files WHERE path = ?", ["x"], 1)
    assert sql == "SELECT * FROM files WHERE path = ?"
    assert params == ["x"]


def test_scoping_reaches_the_real_query_handler():
    """End to end, against a real sqlite database."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE conversations (id INTEGER, user_id INTEGER, title TEXT)")
    connection.executemany(
        "INSERT INTO conversations VALUES (?, ?, ?)",
        [(1, 7, "mine"), (2, 9, "someone else's")])

    class Db:
        """Just enough database for the handler."""
        # Mirrors Database.query_rows — a double whose signature drifts from
        # the real one is how the handler shipped calling a method that did
        # not exist.
        def query_rows(self, sql, params=(), max_rows=500):
            """Run a read."""
            return connection.execute(sql, tuple(params)).fetchmany(max_rows)

    ctx = type("Ctx", (), {"db": Db(), "user_id": 7})()
    result = call_handler(
        R.DB_QUERY, ctx, {"sql": "SELECT * FROM my_conversations"})
    assert result.ok
    assert [row["title"] for row in result.data] == ["mine"]

    refused = call_handler(
        R.DB_QUERY, ctx, {"sql": "SELECT * FROM conversations"})
    assert not refused.ok


def test_the_db_handlers_speak_the_real_database_api(tmp_path):
    """Read, write and DDL against ``pipeline.database.Database`` itself.

    Every double above is a signature the real class may not have. This is
    the one that catches it: all three handlers shipped calling ``query``/
    ``execute_write`` with an argument list neither method accepted, and the
    tool on top of them failed on every statement.
    """
    from pipeline.database import Database

    ctx = SimpleNamespace(db=Database(str(tmp_path / "t.db")), user_id=1)

    assert call_handler(R.DB_QUERY, ctx, {"sql": "SELECT 1 AS one"}).data == [
        {"one": 1}]
    assert call_handler(R.DB_QUERY, ctx,
                        {"sql": "SELECT ? AS given",
                         "params": ["x"]}).data == [
            {"given": "x"}]
    assert call_handler(
        R.DB_QUERY, ctx, {"sql": "PRAGMA table_info(files)"}).ok

    assert call_handler(
        R.DB_DEFINE, ctx,
        {"ddl": "CREATE TABLE plug_x (id INTEGER PRIMARY KEY, v TEXT)"}).ok
    assert call_handler(
        R.DB_WRITE, ctx,
        {"sql": "INSERT INTO plug_x (v) VALUES (?)",
         "params": ["hi"]}).ok
    assert call_handler(
        R.DB_QUERY, ctx, {"sql": "SELECT v FROM plug_x"}).data == [
        {"v": "hi"}]

    # db.query only reads: ``scope_sql`` answers whose rows, never whether the
    # statement mutates, so the kernel-table check lives on the write path and
    # a mutation arriving here must be refused rather than run.
    mutating = call_handler(R.DB_QUERY, ctx, {"sql": "DELETE FROM plug_x"})
    assert not mutating.ok and "db.write" in mutating.error
    assert call_handler(R.DB_QUERY, ctx, {"sql": "SELECT v FROM plug_x"}).data


def test_a_read_is_capped_before_it_crosses(tmp_path):
    """An unbounded SELECT is a hazard, not a result."""
    from pipeline.database import Database
    from sandbox.handlers.kernel import DB_MAX_ROWS

    db = Database(str(tmp_path / "t.db"))
    ctx = SimpleNamespace(db=db, user_id=1)
    call_handler(
        R.DB_DEFINE, ctx, {"ddl": "CREATE TABLE plug_many (n INTEGER)"})
    for n in range(DB_MAX_ROWS + 10):
        call_handler(
            R.DB_WRITE, ctx, {"sql": "INSERT INTO plug_many VALUES (?)",
                                   "params": [n]})

    assert len(call_handler(
        R.DB_QUERY, ctx,
        {"sql": "SELECT n FROM plug_many"}).data) == DB_MAX_ROWS
    assert len(call_handler(
        R.DB_QUERY, ctx,
        {"sql": "SELECT n FROM plug_many", "max_rows": 3}).data) == 3
    # The cap is a ceiling, not a default a caller may raise.
    assert len(call_handler(R.DB_QUERY, ctx, {"sql": "SELECT n FROM plug_many",
              "max_rows": DB_MAX_ROWS * 10}).data) == DB_MAX_ROWS


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Handlers that need the kernel degrade rather than explode.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_a_missing_capability_is_an_ordinary_failure():
    """This is a microkernel: the timekeeper may simply not be installed."""
    ctx = type("Ctx", (), {"services": {}, "db": None, "runtime": None})()
    for kind in (R.CRON_LIST, R.AGENT_COMPLETE, R.DB_QUERY):
        result = call_handler(kind, ctx, {"sql": "select 1"})
        assert not result.ok
        assert "not available" in result.error or "requires" in result.error


def test_asking_what_a_file_is_always_answers():
    """``parse.modality`` is the exception: it is kernel routing, not a service.

    The native-modality defaults cover image/audio/video with no parser
    installed at all, which is what lets attachment routing inline a .png
    into a vision model on a bare install.
    """
    ctx = type("Ctx", (), {"services": {}, "db": None, "runtime": None})()
    result = call_handler(R.PARSE_MODALITY, ctx, {"extension": ".png"})
    assert result.ok
    assert result.data == "image"


def test_password_hash_never_leaves_through_user_read():
    """Hidden at the handler, not only in the query scoper."""
    class Db:
        """A database with a user in it."""
        def get_user(self, uid):
            """Return the row, secrets and all."""
            return {"id": uid, "username": "henry",
                    "password_hash": "$2b$verysecret"}

    ctx = type("Ctx", (), {"db": Db(), "user_id": 1})()
    data = call_handler(R.USER_READ, ctx, {}).data
    assert data["username"] == "henry"
    assert "password_hash" not in data


def test_service_call_respects_exports():
    """Which service methods are reachable is declared, not guessed."""
    class Service:
        """A service with one exported method and one internal one."""
        exports = ["public"]

        def public(self):
            """Reachable."""
            return "yes"

        def internal(self):
            """Not reachable."""
            return "no"

    ctx = type("Ctx", (), {"services": {"svc": Service()}})()
    allowed = call_handler(
        R.SERVICE_CALL, ctx, {"name": "svc", "method": "public"})
    assert allowed.data == "yes"

    refused = call_handler(R.SERVICE_CALL, ctx, {"name": "svc",
                                             "method": "internal"})
    assert refused.denied
    assert "not exported" in refused.error


def test_a_service_that_raises_fails_the_call_only():
    """One bad method is a failed Request, not a broken kernel."""
    class Service:
        """Explodes on demand."""
        exports = ["boom"]

        def boom(self):
            """Fail."""
            raise ValueError("nope")

    ctx = type("Ctx", (), {"services": {"svc": Service()}})()
    result = call_handler(
        R.SERVICE_CALL, ctx, {"name": "svc", "method": "boom"})
    assert not result.ok
    assert "nope" in result.error


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# End to end through the interpreter.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_a_secret_is_usable_without_being_readable(tmp_path):
    """The property the whole mechanism exists for."""
    ctx = type("Ctx", (), {"config": {"secret_brave_api_key": "sk-real"}})()
    interpreter = Interpreter(context=ctx)
    try:
        from sandbox.interpreter import Execution
        execution = Execution(name="probe", chain=Chain().push("probe"))
        sdk = SDK(interpreter.channel(execution))

        seen = sdk.config.read("secret_brave_api_key")
        assert seen == "<secret:secret_brave_api_key>"
        assert "sk-real" not in str(seen)
    finally:
        interpreter.shutdown()


def test_every_namespace_is_exactly_one_request_family():
    """The rule that makes the SDK learnable from the catalogue and back.

    A namespace that merged two families (``sdk.tools.run_command`` running a
    *command*) means the author has to memorise the exceptions rather than
    read one off the other.
    """
    from sandbox.guest.sdk import SDK, _Namespace

    sent = {}

    class Recorder:
        """Notes which family each namespace emits."""

        def __init__(self, name):
            self.name = name

        def send(self, request):
            """Record the family."""
            sent.setdefault(self.name, set()).add(request.family)
            return Result(data=None)

        def log(self, level, message):
            """Ignore."""

    import inspect

    probe = SDK(Recorder("probe"))
    for name in [n for n in vars(probe) if not n.startswith("_")]:
        namespace = getattr(probe, name)
        if not isinstance(namespace, _Namespace):
            continue
        namespace._sdk = SDK(Recorder(name))
        for attr in dir(namespace):
            if attr.startswith("_"):
                continue
            method = getattr(namespace, attr)
            if not inspect.ismethod(method):
                continue
            args = {p.name: "x" for p in
                    inspect.signature(method).parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
            method(**args)

    mixed = {name: families for name, families in sent.items()
             if len(families) > 1}
    assert not mixed, f"namespaces spanning several families: {mixed}"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Plaintext: the limit of handles, and the door through it.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_revealing_a_secret_always_asks():
    """Handles work when the kernel makes the call. A plugin driving a
    foreign library performs its own I/O, so it genuinely needs the value —
    and that is worth a dialog every single time."""
    decision = classify(Request(R.SECRET_REVEAL, {"name": "gmail_secret"}),
                        Chain(root="user"))
    assert not decision.safe
    assert "gmail_secret" in decision.reason


def test_reveal_hands_over_the_real_value():
    """It is a door, not a decoration."""
    ctx = type("Ctx", (), {"config": {"brave_api_key": "sk-real"}})()
    result = call_handler(R.SECRET_REVEAL, ctx, {"name": "brave_api_key"})
    assert result.ok
    assert result.data == "sk-real"


def test_reveal_of_something_absent_fails_cleanly():
    """A missing secret is a failure, not an empty string quietly used."""
    ctx = type("Ctx", (), {"config": {}})()
    result = call_handler(R.SECRET_REVEAL, ctx, {"name": "nope"})
    assert not result.ok


def test_the_dialog_says_plainly_what_reveal_means():
    """The user has to understand they are handing over the value itself."""
    from sandbox.approval import describe

    request = Request(R.SECRET_REVEAL, {"name": "gmail_client_secret"})
    title, body = describe(Chain(root="user").push("service_gmail"), request,
                           classify(request, Chain(root="user")))
    # Title and body are rendered together by every frontend; what matters is
    # that the pair says it, and that the word survives wherever it lives.
    assert "plaintext" in f"{title}\n{body}".lower()
    assert "hold it directly" in body
    assert "gmail_client_secret" in body
    assert "Asked by service_gmail" in body


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Knowing which settings are secrets.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_the_prefix_is_the_declaration():
    """A config setting holding a credential is called secret_something."""
    from sandbox import credentials as S

    assert S.is_secret("secret_brave_api_key")
    assert S.is_secret("secret_notion_integration")   # name says nothing
    assert not S.is_secret("max_tokens")


def test_an_unmarked_setting_is_not_a_secret():
    """Not marked is an answer, not a maybe. The prefix is the whole rule."""
    from sandbox import credentials as S

    assert not S.is_secret("brave_api_key")


def test_environment_variables_are_guessed_because_nothing_declares_them():
    """No plugin owns OPENAI_API_KEY, and its name was chosen elsewhere."""
    from sandbox import credentials as S

    assert S.is_secret("OPENAI_API_KEY", guess=True)
    assert S.is_secret("GMAIL_CLIENT_SECRET", guess=True)
    assert not S.is_secret("PATH", guess=True)
    assert not S.is_secret("max_tokens", guess=True)


def test_env_read_redacts_but_config_read_obeys_the_prefix():
    """The two sources answer to different rules, at the handler."""
    import os

    ctx = type("Ctx", (), {"config": {"secret_brave_key": "sk-a",
                                      "brave_api_key": "sk-b"}})()
    read = partial(call_handler, R.CONFIG_READ)
    assert read(ctx, {"key": "secret_brave_key"}).data == "<secret:secret_brave_key>"
    assert read(ctx, {"key": "brave_api_key"}).data == "sk-b"

    os.environ["SB_TEST_API_KEY"] = "sk-env"
    try:
        out = call_handler(R.ENV_READ, ctx, {"name": "SB_TEST_API_KEY"})
        assert out.data == "<secret:SB_TEST_API_KEY>"
    finally:
        os.environ.pop("SB_TEST_API_KEY", None)


def test_the_validator_flags_an_unmarked_credential(tmp_path):
    """The heuristic stops being policy and becomes a warning to the author."""
    from sandbox.validator import NOTE, validate_file

    plugin = tmp_path / "tool_search.py"
    plugin.write_text('''"""A tool."""

from guest.bases import BaseTool


class Search(BaseTool):
    """Search the web."""

    name = "search"
    config_settings = [
        ("Brave key", "brave_api_key", "The key.", "", {}),
        ("Result count", "brave_results", "How many.", 5, {}),
    ]
''', encoding="utf-8")

    report = validate_file(plugin)
    assert report.ok          # a warning to the author, not a refusal
    notes = " ".join(f.message + " " + f.fix for f in report.of(NOTE))
    assert "brave_api_key" in notes
    assert "secret_brave_api_key" in notes
    assert "brave_results" not in notes


def test_an_already_marked_setting_is_not_flagged(tmp_path):
    """Doing it right must be silent, or the warning becomes noise."""
    from sandbox.validator import NOTE, validate_file

    plugin = tmp_path / "tool_search.py"
    plugin.write_text('''"""A tool."""

from guest.bases import BaseTool


class Search(BaseTool):
    """Search the web."""

    name = "search"
    config_settings = [
        ("Brave key", "secret_brave_api_key", "The key.", "", {}),
    ]
''', encoding="utf-8")

    assert not validate_file(plugin).of(NOTE)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Parsing: what can leave the parser, and what cannot.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _register(output, modality="text", success=True, error="",
              also_contains=()):
    """Register a real parser for .probe and return the handler's answer."""
    import parsing
    from parsing import ParseResult

    calls = []

    def parser(sdk, path, config=None):
        """Stand in for a real parse_*.py helper."""
        calls.append(path)
        return ParseResult(modality=modality, success=success, error=error,
                           output=output, also_contains=list(also_contains))

    parsing.register([".probe"], modality, parser)
    return calls


def _parse(modality="text"):
    """Run the parse.file handler the way sandboxed code reaches it."""
    from types import SimpleNamespace

    ctx = SimpleNamespace(services={})
    return call_handler(R.PARSE_FILE, ctx, {"path": "notes.probe",
                                        "modality": modality})


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is module-global; a leaked parser hides a real failure."""
    import parsing

    yield
    parsing.clear()


def test_parsing_returns_the_output_not_the_result_object():
    """``ParseResult`` has ``output``; there has never been a ``text``.

    The old handler reached for ``.text`` and fell through to the whole
    ParseResult, which looked right in-process and could not be serialized
    at all through a subprocess.
    """
    _register("hello world")
    result = _parse()
    assert result.ok
    assert result.data == "hello world"


def test_parsing_carries_multi_modal_discovery():
    """``also_contains`` is how the pipeline learns a PDF holds images."""
    _register("text", also_contains=["image", "tabular"])
    assert _parse().also_contains == ["image", "tabular"]


def test_a_failed_parse_is_a_failed_request():
    """A parse that did not work must not read as an empty success."""
    _register(None, success=False, error="not readable")
    result = _parse()
    assert not result.ok
    assert "not readable" in result.error


@pytest.mark.parametrize("modality", ["image", "audio", "video", "tabular"])
def test_live_object_modalities_are_refused_with_an_explanation(modality):
    """These resolve to PIL images, numpy arrays, an open av.Container.

    None of them can cross, and handing back a broken object would be worse
    than saying so — the message has to point at the way that does work.
    """
    calls = _register(object(), modality=modality)
    result = _parse(modality)
    assert not result.ok
    assert "own box" in result.error
    # Refused before the parser ran: no work done, no live object created.
    assert calls == []


@pytest.mark.parametrize("modality", ["text", "container"])
def test_crossable_modalities_are_allowed(modality):
    """Text and extracted paths are what the rest of the system consumes."""
    payload = "words" if modality == "text" else ["/tmp/a.png"]
    calls = _register(payload, modality=modality)
    result = _parse(modality)
    assert result.ok
    assert result.data == payload
    assert calls == ["notes.probe"]


def test_a_service_listing_redacts_like_every_other_listing():
    """``/services`` printed a provider's API key in plaintext into the chat.

    ``_service_list`` was the one ``details=True`` handler that did not
    redact — frontends, tools, tasks and ``config.read`` all did. Masking has
    to happen here rather than in the command, because a command-side mask
    still puts the key on the wire and into the ledger.
    """
    from sandbox.handlers.kernel import _service_list

    class Service:
        """A service declaring one credential and one ordinary setting."""
        loaded = True
        config_settings = [
            ("Brave Key", "secret_brave_api_key", "the key", "", {}),
            ("Result count", "search_count", "how many", 5, {}),
        ]

    class Ctx:
        """Kernel context holding the plaintext, as the real one does."""
        services = {"web_search_provider": Service()}
        config = {"secret_brave_api_key": "sk-real-key",
                  "search_count": 5}
        user_id = 1
        db = None

    rows = _service_list(Ctx(), {"details": True}).data
    settings = {s["key"]: s["current"] for s in rows[0]["config_settings"]}

    assert settings["secret_brave_api_key"] == "<secret:secret_brave_api_key>"
    assert "sk-real-key" not in str(rows)
    # Ordinary settings are untouched; this is redaction, not blanking.
    assert settings["search_count"] == 5


# ──────────────────────────────────────────────────────────────────────
# What a config.write dialog actually asks.
# ──────────────────────────────────────────────────────────────────────

def test_a_config_dialog_names_the_value_not_just_the_key():
    """"Change setting net_allowed_hosts" is not a question anyone can answer.

    The egress allowlist is the user's, deliberately — a plugin cannot declare
    its own reach. But a plugin *proposing* a change is ordinary
    ``config.write``, and the dialog is the whole of the consent. Naming only
    the key hid the entire decision: which hosts.
    """
    from sandbox.policy import describe_setting_change

    chain = Chain(root="service_web_search").push("web_search")
    reason = classify(Request(R.CONFIG_WRITE, {
        "key": "net_allowed_hosts",
        "value": ["api.search.brave.com", "html.duckduckgo.com"]}), chain).reason

    assert "api.search.brave.com" in reason
    assert "html.duckduckgo.com" in reason

    # Numbers read as numbers, absence reads as clearing.
    assert describe_setting_change("data_retention_days", 30) == (
        "set data_retention_days to 30")
    assert describe_setting_change("anything", None) == "clear setting anything"
    # A long list is summarised rather than cut mid-hostname.
    many = describe_setting_change(
        "net_allowed_hosts", [f"h{i}.example.com" for i in range(40)])
    assert many == "set net_allowed_hosts to [40 entries]"


def test_the_dialog_does_not_become_a_way_to_read_a_secret():
    """Asking permission to *write* a credential must not print it."""
    from sandbox.policy import describe_setting_change

    shown = describe_setting_change("secret_brave_api_key", "sk-real-key")

    assert "sk-real-key" not in shown
    assert "<secret:secret_brave_api_key>" in shown
