"""Canvas state machine — parallel to the conversation state machine.

Holds canvas state (layers, palette, size, history) and routes discrete
actions (add layer, remove, move, change controls, regenerate, clear)
through a single ``cs.enact(...)`` site. Mirrors the
ConversationState/Action/ActionMap/Runtime pattern but stripped to the
minimum: one phase, no participants, no forms, no persistence, no
rendering.

Old canvas code in state_machine/action.py and runtime/conversation_runtime.py
continues to run untouched. This package is dormant infrastructure until a
future swap wires it into SecondBrainContext and the frontends.
"""

from canvas.state import CANVAS_IDLE, CanvasState
from canvas.runtime import CanvasRuntime

__all__ = ["CanvasState", "CanvasRuntime", "CANVAS_IDLE"]
