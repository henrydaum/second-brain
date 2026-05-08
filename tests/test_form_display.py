from types import SimpleNamespace

from plugins.frontends.frontend_repl import ReplFrontend
from plugins.frontends.frontend_telegram import TelegramFrontend
from state_machine.conversation import CallableSpec, ConversationState, FormStep, Participant
from state_machine.form_display import form_step_display


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
    assert "JSON array" in display["assist"]
    assert "/skip" in display["assist"]


def test_boolean_display_uses_true_false_choices():
    display = form_step_display(FormStep("run_immediately", "Run immediately?", False, "boolean", default=False))

    assert display["choices"] == [{"value": True, "label": "True"}, {"value": False, "label": "False"}]
    assert display["assist"] == "Select an option. Send /skip to use the default: False."


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


def test_telegram_form_prompt_omits_command_header_and_raw_type():
    form = {
        "name": "agent",
        "field": FormStep("profile_name", "Select an agent profile.", True, enum=["default"]).to_dict(),
        "display": form_step_display(FormStep("profile_name", "Select an agent profile.", True, enum=["default"])),
    }

    text = TelegramFrontend()._prompt(form)

    assert "Select an agent profile." in text
    assert not text.startswith("<b>agent</b>")
    assert "Type:" not in text
    assert "/cancel" not in text


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
