"""Ask the active user for typed input through the approval dialog flow."""

from plugins.BaseTool import BaseTool, ToolResult
from state_machine.conversation import FormStep
from state_machine.form_display import form_step_display

TYPES = {"string", "integer", "int", "number", "boolean", "array", "object"}


class AskUserQuestion(BaseTool):
    name = "ask_user_question"
    description = (
        "Ask the user a question and wait for a typed answer. Use this when the "
        "agent needs user input before continuing. Supports strings, integers, "
        "numbers, booleans, arrays, objects, and enum choices. Cancel or timeout "
        "returns a failed result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question to show the user."},
            "title": {"type": "string", "description": "Short dialog title. Defaults to 'Question for you'."},
            "type": {"type": "string", "enum": sorted(TYPES), "description": "Expected answer type. Defaults to string."},
            "enum": {"type": "array", "items": {}, "description": "Allowed choices. If provided, the answer must be one of these values."},
            "default": {"description": "Default value for optional blank answers."},
            "required": {"type": "boolean", "description": "Whether an answer is required. Defaults to true."},
            "timeout": {"type": "integer", "description": "Seconds to wait before cancelling. Defaults to 300, max 3600."},
        },
        "required": ["question"],
    }
    max_calls = 5
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        ask = getattr(context, "request_user_input", None)
        if ask is None:
            return ToolResult.failed("User input is not available — no live session is configured.")
        question = (kwargs.get("question") or "").strip()
        if not question:
            return ToolResult.failed("question is required.")
        type_ = (kwargs.get("type") or "string").strip().lower()
        if type_ not in TYPES:
            return ToolResult.failed(f"type must be one of: {', '.join(sorted(TYPES))}.")
        try:
            timeout = min(max(int(kwargs.get("timeout", 300)), 1), 3600)
        except (TypeError, ValueError):
            return ToolResult.failed("timeout must be an integer number of seconds.")
        title = (kwargs.get("title") or "Question for you").strip() or "Question for you"
        display = form_step_display(FormStep("answer", question, kwargs.get("required", True), type_, kwargs.get("enum"), default=kwargs.get("default")))
        prompt = "\n\n".join(part for part in [display["prompt"], display.get("assist")] if part)
        req = ask(
            title,
            prompt,
            type=type_,
            enum=kwargs.get("enum"),
            default=kwargs.get("default"),
            required=kwargs.get("required", True),
        )
        if not req.wait(timeout=timeout):
            req.metadata["timed_out"] = True
            if getattr(context, "runtime", None) is not None and getattr(context, "session_key", None):
                context.runtime.handle_action(context.session_key, "cancel")
            return ToolResult.failed("User question timed out.")
        if req.metadata.get("cancelled"):
            return ToolResult.failed("User cancelled the question.")
        return ToolResult(data={"value": req.value, "type": type_, "request_id": req.id}, llm_summary=f"User answered: {req.value!r}")
