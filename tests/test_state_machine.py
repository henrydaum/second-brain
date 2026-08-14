"""Tests for the state-machine layer.

Covers the pure conversation primitives the kernel relies on: schema->form
derivation, form-step display payloads, REPL form rendering, and the
serialization/compaction markers that survive across restarts. The REPL is the
only kernel frontend, so frontend assertions here use it directly.
"""

from types import SimpleNamespace

from plugins.native.frontend import BaseFrontend
from bundled.frontends.helpers.formatters import render_plain
from state_machine.conversation import CallableSpec, ConversationState, FormStep, Participant
from state_machine.form_display import form_step_display
from state_machine.forms import schema_to_form_steps
from state_machine.serialization import (
    latest_state,
    messages_to_history,
    save_compaction_marker,
    save_state_marker,
)


class ReplFrontend(BaseFrontend):
    """Native test surface for host-side input coercion and form projection."""

    name = "repl"

    def render_messages(self, _key, messages):
        for message in messages:
            print(render_plain(message))

    def render_form_field(self, _key, form):
        field = form.get("field") or {}
        display = form.get("display") or {}
        prompt = (
            display.get("prompt") or field.get("prompt")
            or field.get("name") or "Input required"
        )
        print(f"{render_plain(str(prompt))}{self._hints(display or field)}")

    def render_error(self, _key, error):
        print(f"[error] {(error or {}).get('message') or error}")

    render_attachments = render_buttons = render_approval_request = (
        lambda self, *args, **kwargs: None
    )

    @staticmethod
    def _hints(field):
        parts = []
        if field.get("allow_back"):
            parts.append("/back to go back")
        return f" ({'; '.join(parts)})" if parts else ""


# ── Schema -> form derivation ────────────────────────────────────────

def test_schema_form_prompts_include_action_and_description():
    steps = schema_to_form_steps({
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the scheduled conversation."},
            "one_time": {"type": "boolean", "description": "If true, run once at the next cron match.", "default": False},
        },
        "required": ["title"],
    }, prompt_optional=True)

    assert steps[0].prompt == "Enter a title.\nShort title for the scheduled conversation."
    assert steps[1].prompt == "Choose one time.\nIf true, run once at the next cron match."
    assert steps[1].prompt_when_missing is True


# ── Form-step display payloads ───────────────────────────────────────

def test_enum_display_uses_choices_without_raw_type_hint():
    display = form_step_display(FormStep("profile_name", "Select an agent profile.", True, enum=["default", "add"]))

    assert display["prompt"] == "Select an agent profile."
    assert display["assist"] == "Select an option."
    assert display["choices"] == [{"value": "default", "label": "default"}, {"value": "add", "label": "add"}]
    assert display["input_mode"] == "choice"


def test_optional_default_display_includes_skip_guidance():
    display = form_step_display(FormStep("endpoint", "Optional endpoint URL.", False, default=""))

    assert display["allow_skip"] is True
    assert "leave this blank" in display["assist"]
    assert "Type:" not in display["assist"]


def test_json_display_gives_concrete_guidance():
    display = form_step_display(FormStep("tools_list", "Optional tool names.", False, "array", default=[]))

    assert display["input_mode"] == "json"
    assert "Enter a list of items, one on each line" in display["assist"]
    assert "/skip" in display["assist"]


def test_array_form_values_accept_one_item_per_line():
    assert FormStep("tools_list", type="array").coerce("lexical_search\nsemantic_search") == ["lexical_search", "semantic_search"]


def test_array_form_values_wrap_json_scalar():
    assert FormStep("services", type="array").coerce('"web_search_provider"') == ["web_search_provider"]


def test_boolean_display_uses_true_false_choices():
    display = form_step_display(FormStep("one_time", "Run once?", False, "boolean", default=False))

    assert display["choices"] == [{"value": True, "label": "True"}, {"value": False, "label": "False"}]
    assert display["assist"] == "Select an option. Send /skip to use the default: False."


# ── Form lifecycle through ConversationState + dispatch ──────────────

def test_runtime_decorates_form_with_display_payload():
    cs = ConversationState([
        Participant("user", "user", commands={"agent": CallableSpec("agent", form=[FormStep("profile_name", "Select an agent profile.", True, enum=["default"])])}),
    ])

    result = cs.enact("call_command", {"name": "agent", "args": {}}, "user")

    from runtime.dispatch import decorate_form
    from runtime.session import RuntimeResult
    out = RuntimeResult()
    decorate_form(SimpleNamespace(cs=cs), out)
    assert result.ok
    assert out.form["display"]["prompt"] == "Select an agent profile."
    assert out.form["display"]["choices"][0]["value"] == "default"
    assert out.form["display"]["allow_back"] is False


def test_runtime_decorates_form_with_back_available_after_input():
    cs = ConversationState([
        Participant("user", "user", commands={"agent": CallableSpec("agent", form=[FormStep("profile_name", "Select an agent profile.", True), FormStep("action", "Choose.", True)])}),
    ])

    assert cs.enact("call_command", {"name": "agent", "args": {}}, "user").ok
    assert cs.enact("submit_form_text", "default", "user").ok

    from runtime.dispatch import decorate_form
    from runtime.session import RuntimeResult
    out = RuntimeResult()
    decorate_form(SimpleNamespace(cs=cs), out)

    assert out.form["field"]["name"] == "action"
    assert out.form["display"]["allow_back"] is True


# ── Which channel a success one-liner travels on ─────────────────────
#
# A form-navigation acknowledgement is a command reporting on its own progress,
# so it belongs on ``callable_output``. Landing on ``messages`` put a line of
# chat in the transcript every time somebody pressed Back in a settings dialog.
#
# ``cancel`` was held back from that move on the reading that it ends the turn
# and that ``/cancel`` reads its answer. Neither applies to *this* result:
# ``handle_action`` short-circuits every base-phase and busy cancel before
# dispatch, so an action reaching ``Cancel.execute`` always has a frame to pop,
# and ``BaseFrontend.submit_text`` intercepts ``/cancel`` before the command
# ever runs. What was left was a form's own Cancel button typing into the chat.


def _two_step_form():
    return ConversationState([
        Participant("user", "user", commands={"agent": CallableSpec("agent", form=[FormStep("profile_name", "Select an agent profile.", True), FormStep("action", "Choose.", True)])}),
    ])


def test_back_acknowledgement_is_command_output_not_conversation():
    from runtime.session import RuntimeResult

    cs = _two_step_form()
    assert cs.enact("call_command", {"name": "agent", "args": {}}, "user").ok
    assert cs.enact("submit_form_text", "default", "user").ok

    out = RuntimeResult().add_action_result(cs.enact("back_form", None, "user"))

    assert out.callable_output == ["Back."]
    assert out.messages == []


def test_skip_acknowledgement_is_command_output_not_conversation():
    from runtime.session import RuntimeResult

    # A required step, then an optional one that asks to be shown anyway: a
    # required field refuses to be skipped, and an optional field is only
    # *presented* — and so only skippable — with `prompt_when_missing`.
    cs = ConversationState([
        Participant("user", "user", commands={"agent": CallableSpec("agent", form=[FormStep("profile_name", "Select an agent profile.", True), FormStep("notes", "Anything to add?", False, prompt_when_missing=True)])}),
    ])
    assert cs.enact("call_command", {"name": "agent", "args": {}}, "user").ok
    assert cs.enact("submit_form_text", "default", "user").ok

    result = cs.enact("skip_form", None, "user")
    assert result.ok
    out = RuntimeResult().add_action_result(result)

    # Skipping the last field also *runs* the command, so this exercises the
    # path where the acknowledgement is carried by the replayed action's own
    # result rather than by `skip_form`'s.
    assert out.callable_output == ["Skipped."]
    assert out.messages == []


def test_cancel_acknowledgement_is_command_output_not_conversation():
    from runtime.session import RuntimeResult

    cs = _two_step_form()
    assert cs.enact("call_command", {"name": "agent", "args": {}}, "user").ok

    out = RuntimeResult().add_action_result(cs.enact("cancel", None, "user"))

    assert out.callable_output == ["Cancelled."]
    assert out.messages == []


# ── REPL form rendering ──────────────────────────────────────────────

def test_repl_form_rendering_uses_display_payload(capsys):
    form = {
        "name": "agent",
        "field": FormStep("profile_name", "Select an agent profile.", True, enum=["default"]).to_dict(),
        "display": form_step_display(FormStep("profile_name", "Select an agent profile.", True, enum=["default"])),
    }

    ReplFrontend().render_form_field("default", form)
    out = capsys.readouterr().out

    assert "Select an agent profile." in out
    assert "agent:" not in out
    assert "type:" not in out


def test_repl_form_rendering_shows_back_hint_when_available(capsys):
    step = FormStep("action", "Choose.", True, enum=["edit"])
    display = form_step_display(step)
    display["allow_back"] = True

    ReplFrontend().render_form_field("default", {"field": step.to_dict(), "display": display})
    out = capsys.readouterr().out

    assert "/back to go back" in out


def test_frontend_text_back_submits_back_form_action():
    cs = ConversationState([
        Participant("user", "user", commands={"agent": CallableSpec("agent", form=[FormStep("profile_name", "Select an agent profile.", True), FormStep("action", "Choose.", True)])}),
    ])
    assert cs.enact("call_command", {"name": "agent", "args": {}}, "user").ok
    assert cs.enact("submit_form_text", "default", "user").ok
    seen = []
    repl = ReplFrontend()
    repl.runtime = SimpleNamespace(get_session=lambda _key: SimpleNamespace(cs=cs))
    repl.submit = lambda _key, action_type, payload=None: seen.append((action_type, payload))

    repl.submit_text("default", "/back")

    assert seen == [("back_form", None)]


def test_frontend_unknown_slash_command_does_not_submit(capsys):
    cs = ConversationState([Participant("user", "user", commands={})])
    seen = []
    repl = ReplFrontend()
    repl.runtime = SimpleNamespace(get_session=lambda _key: SimpleNamespace(cs=cs))
    repl.commands = SimpleNamespace(all_commands=lambda: [])
    repl.submit = lambda _key, action_type, payload=None: seen.append((action_type, payload))

    result = repl.submit_text("default", "/doctor")
    out = capsys.readouterr().out

    assert "`/doctor` isn't a recognized slash command." in out
    assert result.ok is False
    assert seen == []


def test_frontend_busy_session_does_not_parse_slash_command_args():
    cs = ConversationState([Participant("user", "user", commands={})])
    seen = []
    repl = ReplFrontend()
    repl.runtime = SimpleNamespace(get_session=lambda _key: SimpleNamespace(cs=cs, busy=True))
    repl.commands = SimpleNamespace(parse_args=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not parse while busy")))
    repl.submit = lambda _key, action_type, payload=None: seen.append((action_type, payload))

    repl.submit_text("default", "/packages uninstall all-parsers")

    assert seen == [("send_text", "/packages uninstall all-parsers")]


def test_frontend_unknown_slash_command_in_form_does_not_submit(capsys):
    cs = ConversationState([
        Participant("user", "user", commands={"agent": CallableSpec("agent", form=[FormStep("profile_name", "Select an agent profile.", True)])}),
    ])
    assert cs.enact("call_command", {"name": "agent", "args": {}}, "user").ok
    seen = []
    repl = ReplFrontend()
    repl.runtime = SimpleNamespace(get_session=lambda _key: SimpleNamespace(cs=cs))
    repl.commands = SimpleNamespace(all_commands=lambda: [])
    repl.submit = lambda _key, action_type, payload=None: seen.append((action_type, payload))

    result = repl.submit_text("default", "/doctor")
    out = capsys.readouterr().out

    assert "`/doctor` isn't a recognized slash command." in out
    assert result.ok is False
    assert seen == []


# ── Compaction / state markers ───────────────────────────────────────

class _FakeDb:
    """Minimal message store recording rows the way the real DB does."""

    def __init__(self):
        self.rows = []

    def save_message(self, conversation_id, role, content, tool_call_id=None, tool_name=None, attachments=None, author=None):
        self.rows.append({"role": role, "content": content, "tool_call_id": tool_call_id, "tool_name": tool_name, "attachments": attachments or [], "author": author})


def test_compaction_marker_preserves_db_rows_but_hides_pre_checkpoint_replay():
    db = _FakeDb()
    db.save_message(1, "user", "old user")
    db.save_message(1, "assistant", "old assistant")
    save_compaction_marker(db, 1, "Earlier summary.")
    db.save_message(1, "user", "after")
    save_state_marker(db, 1, {"active_agent_profile": "builder"})

    history = messages_to_history(db.rows)

    assert [r["content"] for r in db.rows if r["role"] != "system"] == ["old user", "old assistant", "after"]
    # The bridge pair is authored: it is the kernel handing the model a summary,
    # not two turns anybody took. "after" is the person, so it carries nothing.
    assert history == [
        {"role": "user", "author": "compaction", "content": "[Conversation summary from earlier]\nEarlier summary."},
        {"role": "assistant", "author": "compaction", "content": "Understood - I have the earlier context."},
        {"role": "user", "content": "after"},
    ]
    assert latest_state(db.rows)["active_agent_profile"] == "builder"
