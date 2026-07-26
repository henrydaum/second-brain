"""The Request catalogue wired to handlers: completeness, secrets, scoping.

Three properties matter more than any individual handler:

- every Request in the catalogue is *classified* — nothing reaches an effect
  without a decision
- every Request is either serviced or explicitly listed as unwired
- the SDK can actually reach everything the catalogue defines
"""

import sqlite3

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Chain, Interpreter, Request, Result
from sandbox.guest import requests as R
from sandbox.guest.sdk import SDK
from sandbox.handlers import HANDLERS, UNWIRED
from sandbox.policy import classify
from sandbox.secrets import handle_for, is_secret, redact, resolve
from sandbox.users import ScopeError, scope_sql


# ──────────────────────────────────────────────────────────────────────
# Completeness.
# ──────────────────────────────────────────────────────────────────────

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


def test_only_plugin_lifecycle_is_still_unwired():
    """The deferred set is deliberate, and it is the one we chose to defer."""
    assert set(UNWIRED) <= {
        R.PLUGIN_REGISTER, R.PLUGIN_UNREGISTER, R.PLUGIN_RELOAD,
        R.PLUGIN_INSTALL, R.PLUGIN_UNINSTALL, R.SERVICE_LOAD, R.SERVICE_UNLOAD,
        R.AGENT_SPAWN, R.AGENT_SCHEDULE,
    }


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


# ──────────────────────────────────────────────────────────────────────
# Policy, family by family.
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Secret handles.
# ──────────────────────────────────────────────────────────────────────

def test_credential_names_are_recognised():
    """Generous on purpose: a false positive costs plaintext nobody needed."""
    for name in ("brave_api_key", "OPENAI_API_KEY", "client_secret",
                 "db_password", "access_token"):
        assert is_secret(name), name
    for name in ("model", "max_tokens", "db_path"):
        assert not is_secret(name), name


def test_a_secret_reads_back_as_a_handle():
    """The sandbox cannot leak what it was never given."""
    assert redact("brave_api_key", "sk-real") == "<secret:brave_api_key>"
    assert redact("model", "opus") == "opus"


def test_handles_resolve_on_the_way_out():
    """Substitution happens in the handler, after policy has decided."""
    lookup = {"brave_api_key": "sk-real"}.get
    payload = {"headers": {"X-Key": handle_for("brave_api_key")},
               "list": [handle_for("brave_api_key")]}
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
    ctx = type("Ctx", (), {"config": {"brave_api_key": "sk-real",
                                      "model": "opus"}})()
    handler = HANDLERS[R.CONFIG_READ]
    assert handler(ctx, {"key": "brave_api_key"}).data == "<secret:brave_api_key>"
    assert handler(ctx, {"key": "model"}).data == "opus"
    everything = handler(ctx, {"key": None}).data
    assert everything["brave_api_key"] == "<secret:brave_api_key>"


# ──────────────────────────────────────────────────────────────────────
# Per-user scoping.
# ──────────────────────────────────────────────────────────────────────

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
        def query(self, sql, params):
            """Run a read."""
            return connection.execute(sql, params).fetchall()

    ctx = type("Ctx", (), {"db": Db(), "user_id": 7})()
    result = HANDLERS[R.DB_QUERY](ctx, {"sql": "SELECT * FROM my_conversations"})
    assert result.ok
    assert [row["title"] for row in result.data] == ["mine"]

    refused = HANDLERS[R.DB_QUERY](ctx, {"sql": "SELECT * FROM conversations"})
    assert not refused.ok


# ──────────────────────────────────────────────────────────────────────
# Handlers that need the kernel degrade rather than explode.
# ──────────────────────────────────────────────────────────────────────

def test_a_missing_capability_is_an_ordinary_failure():
    """This is a microkernel: the timekeeper may simply not be installed."""
    ctx = type("Ctx", (), {"services": {}, "db": None, "runtime": None})()
    for kind in (R.CRON_LIST, R.PARSE_MODALITY, R.AGENT_COMPLETE, R.DB_QUERY):
        result = HANDLERS[kind](ctx, {"sql": "select 1"})
        assert not result.ok
        assert "not available" in result.error or "requires" in result.error


def test_password_hash_never_leaves_through_user_read():
    """Hidden at the handler, not only in the query scoper."""
    class Db:
        """A database with a user in it."""
        def get_user(self, uid):
            """Return the row, secrets and all."""
            return {"id": uid, "username": "henry",
                    "password_hash": "$2b$verysecret"}

    ctx = type("Ctx", (), {"db": Db(), "user_id": 1})()
    data = HANDLERS[R.USER_READ](ctx, {}).data
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
    allowed = HANDLERS[R.SERVICE_CALL](ctx, {"name": "svc", "method": "public"})
    assert allowed.data == "yes"

    refused = HANDLERS[R.SERVICE_CALL](ctx, {"name": "svc",
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
    result = HANDLERS[R.SERVICE_CALL](ctx, {"name": "svc", "method": "boom"})
    assert not result.ok
    assert "nope" in result.error


# ──────────────────────────────────────────────────────────────────────
# End to end through the interpreter.
# ──────────────────────────────────────────────────────────────────────

def test_a_secret_is_usable_without_being_readable(tmp_path):
    """The property the whole mechanism exists for."""
    ctx = type("Ctx", (), {"config": {"brave_api_key": "sk-real"}})()
    interpreter = Interpreter(context=ctx)
    try:
        from sandbox.interpreter import Execution
        execution = Execution(name="probe", chain=Chain().push("probe"))
        sdk = SDK(interpreter.channel(execution))

        seen = sdk.config.read("brave_api_key")
        assert seen.data == "<secret:brave_api_key>"
        assert "sk-real" not in str(seen.data)
    finally:
        interpreter.shutdown()
