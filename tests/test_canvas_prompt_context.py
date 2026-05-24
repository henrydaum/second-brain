from types import SimpleNamespace

from agent.system_prompt import build_prompt_sections
from canvas.runtime import CanvasRuntime
from runtime.runtime_config import session_system_prompt
from runtime.session import RuntimeSession
from state_machine.conversation import ConversationState, Participant


class Registry:
    tools = {"execute_skill": SimpleNamespace(max_calls=3)}

    def get_all_schemas(self):
        return [{"function": {"name": "execute_skill", "description": "Execute canvas skill."}}]


def sectioned_prompt():
    return [
        {"role": "system", "content": "[STATIC SYSTEM PROMPT]\nbase"},
        {"role": "system", "content": "[SEMI-STABLE TOOL/SCHEMA INFO]\ntools"},
        {"role": "system", "content": "[DYNAMIC RUNTIME CONTEXT]\nbase dynamic"},
    ]


def test_session_system_prompt_includes_live_canvas_state():
    canvas = CanvasRuntime()
    cs = canvas.for_session("chat")
    canvas.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "color_field", "kind": "background", "controls": {"mode": "linear"}})
    runtime = SimpleNamespace(db=None, config={}, services={"canvas": canvas}, tool_registry=Registry(), system_prompt=sectioned_prompt, active_session_key="chat")

    dynamic = session_system_prompt(runtime, RuntimeSession("chat", ConversationState([Participant("user", "user"), Participant("agent", "agent")])))()[-1]["content"]

    assert "## Current canvas" in dynamic
    assert "- 0: color_field (background)" in dynamic
    assert '"mode": "linear"' in dynamic


def test_build_prompt_sections_canvas_state_reflects_manual_canvas_edits():
    canvas = CanvasRuntime()
    cs = canvas.for_session("chat")
    canvas.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "gradient_field", "kind": "background", "controls": {"mode": "radial"}})

    before = build_prompt_sections(None, None, Registry(), {"canvas": canvas}, session_key="chat")[-1]["content"]
    canvas.handle_action(cs.canvas_id, "clear", {})
    after = build_prompt_sections(None, None, Registry(), {"canvas": canvas}, session_key="chat")[-1]["content"]

    assert "- 0: gradient_field (background)" in before
    assert "Layers: none. The canvas is empty." in after
    assert "gradient_field" not in after


def test_canvas_prompt_does_not_create_unbound_canvas():
    canvas = CanvasRuntime()

    build_prompt_sections(None, None, Registry(), {"canvas": canvas}, session_key="chat")

    assert canvas.canvases == {}
