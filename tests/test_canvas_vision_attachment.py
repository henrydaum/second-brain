from types import SimpleNamespace

from canvas.runtime import CanvasRuntime
import canvas.render as canvas_render
from attachments.attachment import Attachment, AttachmentBundle
from runtime.conversation_loop import ConversationLoop
from state_machine.conversation import ConversationState, Participant


class FakeLLM:
    context_size = 0

    def __init__(self):
        self.seen = []

    def has_capability(self, name):
        return name == "image"

    def chat_with_tools(self, messages, tools, attachments=None):
        self.seen.append((messages, tools, attachments))
        return SimpleNamespace(content="saw canvas", has_tool_calls=False, tool_calls=[], is_error=False, prompt_tokens=0)


class Registry:
    max_tool_calls = 1
    tools = {}

    def get_all_schemas(self):
        return []


def test_current_canvas_png_attaches_to_vision_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(canvas_render, "RENDERS_DIR", tmp_path)
    canvas = CanvasRuntime()
    cs = canvas.for_session("chat")
    canvas.handle_action(cs.canvas_id, "add_layer", {"technique_slug": "color_field", "kind": "background"})
    cs.render_seed = 123
    image_path = canvas_render.folder_for(cs.canvas) / "123.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    llm = FakeLLM()
    state = ConversationState([Participant("user", "user"), Participant("agent", "agent")], "agent")

    ConversationLoop(llm, Registry(), {}, "", runtime=SimpleNamespace(services={"canvas": canvas}), session_key="chat").drive(state, "agent", [{"role": "user", "content": "look"}])

    attachment = list(llm.seen[0][2])[0]
    assert attachment.path == str(image_path)
    assert attachment.modality == "image"
    assert attachment.metadata["llm_context"] == "The attached image is the current canvas PNG."


def test_native_image_attachment_can_still_add_context_text():
    paths, suffix = AttachmentBundle([Attachment("canvas.png", "png", "current_canvas.png", "image", metadata={"llm_context": "The attached image is the current canvas PNG."})]).for_llm({"image": True})

    assert paths == ["canvas.png"]
    assert suffix == "The attached image is the current canvas PNG."
