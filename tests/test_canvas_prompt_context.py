from types import SimpleNamespace

from agent.system_prompt import build_prompt_sections
from canvas.runtime import CanvasRuntime
from runtime.runtime_config import session_system_prompt
from runtime.session import RuntimeSession
from state_machine.conversation import ConversationState, Participant


class Registry:
    def __init__(self, *names):
        self.tools = {name: SimpleNamespace(max_calls=3) for name in (names or ("execute_technique",))}

    def get_all_schemas(self):
        return [{"function": {"name": name, "description": f"{name}."}} for name in self.tools]


def sectioned_prompt():
    return [{"role": "system", "content": "base\n\ntools\n\nbase dynamic"}]


def test_session_system_prompt_includes_live_canvas_state():
    canvas = CanvasRuntime()
    cs = canvas.for_session("chat")
    canvas.handle_action(cs.canvas_id, "add_layer", {"technique_slug": "color_field", "kind": "background", "controls": {"mode": "linear"}})
    runtime = SimpleNamespace(db=None, config={}, services={"canvas": canvas}, tool_registry=Registry(), system_prompt=sectioned_prompt, active_session_key="chat")

    dynamic = session_system_prompt(runtime, RuntimeSession("chat", ConversationState([Participant("user", "user"), Participant("agent", "agent")])))()[-1]["content"]

    assert "## Current canvas" in dynamic
    assert "Layer count: 1/6" in dynamic
    assert "- index 0: slug=color_field kind=background" in dynamic
    assert '"mode": "linear"' in dynamic


def test_build_prompt_sections_canvas_state_reflects_manual_canvas_edits():
    canvas = CanvasRuntime()
    cs = canvas.for_session("chat")
    canvas.handle_action(cs.canvas_id, "add_layer", {"technique_slug": "gradient_field", "kind": "background", "controls": {"mode": "radial"}})

    before = build_prompt_sections(None, None, Registry(), {"canvas": canvas}, session_key="chat")[-1]["content"]
    canvas.handle_action(cs.canvas_id, "clear", {})
    after = build_prompt_sections(None, None, Registry(), {"canvas": canvas}, session_key="chat")[-1]["content"]

    assert "Use these zero-based indices for manage_layers" in before
    assert "- index 0: slug=gradient_field kind=background" in before
    assert "Layers: none. The canvas is empty." in after
    assert "gradient_field" not in after


def test_canvas_prompt_does_not_create_unbound_canvas():
    canvas = CanvasRuntime()

    build_prompt_sections(None, None, Registry(), {"canvas": canvas}, session_key="chat")

    assert canvas.canvases == {}


def test_prompt_does_not_claim_technique_authoring_without_authoring_tools():
    prompt = build_prompt_sections(None, None, Registry("search_techniques", "read_technique", "execute_technique"), {}, session_key="chat")[-1]["content"]

    assert "Technique authoring is not enabled for this session" in prompt
    assert "create_technique` with a complete" not in prompt


def test_prompt_claims_technique_authoring_when_tool_is_visible():
    prompt = build_prompt_sections(None, None, Registry("search_techniques", "read_technique", "read_technique_guide", "execute_technique", "create_technique"), {}, session_key="chat")[-1]["content"]

    assert "Technique authoring is enabled for this session" in prompt
    assert "create_technique" in prompt
