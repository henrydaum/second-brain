"""State-machine support for action map."""

from state_machine.action import (
    AnswerApproval,
    CallCommand,
    CallTool,
    Cancel,
    BackForm,
    DeleteLayer,
    EndTurn,
    InvalidAction,
    MoveLayer,
    RegenerateCanvas,
    ResetCanvas,
    RunSkill,
    SendAttachment,
    SendText,
    SetCanvasSize,
    SetPalette,
    SetSkillControl,
    SkipForm,
    SubmitFormText,
)
from state_machine.conversation_phases import (
    PHASE_APPROVING_REQUEST,
    PHASE_AWAITING_INPUT,
    PHASE_FILLING_COMMAND_FORM,
    PHASE_FILLING_TOOL_FORM,
    PHASE_PARSING_ATTACHMENT,
)


ACTION_SEND_TEXT = "send_text"
ACTION_CALL_COMMAND = "call_command"
ACTION_CALL_TOOL = "call_tool"
ACTION_SEND_ATTACHMENT = "send_attachment"
ACTION_END_TURN = "end_turn"
ACTION_CANCEL = "cancel"
ACTION_ANSWER_APPROVAL = "answer_approval"
ACTION_SUBMIT_FORM_TEXT = "submit_form_text"
ACTION_SKIP_FORM = "skip_form"
ACTION_BACK_FORM = "back_form"
ACTION_RUN_SKILL = "run_skill"
ACTION_SET_CANVAS_PALETTE = "set_canvas_palette"
ACTION_SET_SKILL_CONTROL = "set_skill_control"
ACTION_REGENERATE_CANVAS = "regenerate_canvas"
ACTION_DELETE_CANVAS_LAYER = "delete_canvas_layer"
ACTION_MOVE_CANVAS_LAYER = "move_canvas_layer"
ACTION_SET_CANVAS_SIZE = "set_canvas_size"
ACTION_RESET_CANVAS = "reset_canvas"

_CANVAS_ACTIONS = {
    ACTION_RUN_SKILL: RunSkill,
    ACTION_SET_CANVAS_PALETTE: SetPalette,
    ACTION_SET_SKILL_CONTROL: SetSkillControl,
    ACTION_REGENERATE_CANVAS: RegenerateCanvas,
    ACTION_DELETE_CANVAS_LAYER: DeleteLayer,
    ACTION_MOVE_CANVAS_LAYER: MoveLayer,
    ACTION_SET_CANVAS_SIZE: SetCanvasSize,
    ACTION_RESET_CANVAS: ResetCanvas,
}

ACTION_MAP = {
    PHASE_AWAITING_INPUT: {
        ACTION_SEND_TEXT: SendText,
        ACTION_CALL_COMMAND: CallCommand,
        ACTION_CALL_TOOL: CallTool,
        ACTION_SEND_ATTACHMENT: SendAttachment,
        ACTION_END_TURN: EndTurn,
        **_CANVAS_ACTIONS,
    },
    PHASE_FILLING_COMMAND_FORM: {ACTION_CALL_COMMAND: CallCommand, ACTION_SEND_TEXT: SubmitFormText, ACTION_SUBMIT_FORM_TEXT: SubmitFormText, ACTION_SKIP_FORM: SkipForm, ACTION_BACK_FORM: BackForm, ACTION_CANCEL: Cancel},
    PHASE_FILLING_TOOL_FORM: {ACTION_CALL_COMMAND: CallCommand, ACTION_SEND_TEXT: SubmitFormText, ACTION_SUBMIT_FORM_TEXT: SubmitFormText, ACTION_SKIP_FORM: SkipForm, ACTION_BACK_FORM: BackForm, ACTION_CANCEL: Cancel},
    PHASE_APPROVING_REQUEST: {ACTION_SEND_TEXT: AnswerApproval, ACTION_ANSWER_APPROVAL: AnswerApproval, ACTION_CANCEL: Cancel},
    PHASE_PARSING_ATTACHMENT: {ACTION_CANCEL: Cancel},
}


def create_action(cs, action_type, content=None, actor_id=None):
    """Create action."""
    return ACTION_MAP.get(cs.phase, {}).get(action_type, InvalidAction)(cs, actor_id, content)


def legal_actions_in_phase(phase: str) -> list[str]:
    """Return the action_types registered for `phase`.

    Mirrors PokerMonster's `display_actions`: a frontend can call this to
    render only the buttons/options that the state machine will accept right
    now, instead of letting the user pick something that will fail
    `is_legal()` after the fact.
    """
    return list(ACTION_MAP.get(phase, {}).keys())
