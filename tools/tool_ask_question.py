"""Ask the active user for typed input.

Sandboxed, and renamed from ``ask_user_question`` — the tool is offered to a
model in a list where every entry is something *it* does, so "user" was
carrying no information.

Almost all of the old body moved into the kernel. Assembling the prompt from a
``FormStep``, waiting on the request, distinguishing a timeout from a
cancellation: all of it now happens behind ``ui.ask``, because the guest cannot
import ``state_machine`` and should not be reimplementing the state machine's
own presentation rules. What is left is the parameter schema and the enum
hygiene below, which is model-facing input handling rather than kernel policy.
"""

dependencies_files = []
dependencies_pip = []
requests = ["ui.ask"]

from guest.bases import BaseTool

TYPES = {"string", "integer", "int", "number", "boolean", "array", "object"}

DEFAULT_TIMEOUT = 300
MAX_TIMEOUT = 3600


class AskQuestion(BaseTool):
    """Ask question."""
    name = "ask_question"
    description = (
        "Ask the user a question and wait for a typed answer. Use this when you "
        "need user input before continuing. Cancel or timeout returns a failed result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question to show the user."},
            "title": {"type": "string", "description": "Short dialog title. Defaults to 'Question for you'."},
            "type": {"type": "string", "enum": sorted(TYPES), "description": "Expected answer type. Defaults to string."},
            "enum": {"type": "array", "items": {"type": "string"}, "description": "Allowed choices as plain non-empty strings. If provided, the answer must be one of these values."},
            "default": {"description": "Default value for optional blank answers."},
            "required": {"type": "boolean", "description": "Whether an answer is required. Defaults to true."},
            "timeout": {"type": "integer", "description": "Seconds to wait before cancelling. Defaults to 300, max 3600."},
        },
        "required": ["question"],
    }
    requires_services = []
    # There is nobody to answer in a background session. The policy function
    # refuses ui.ask unattended anyway; this keeps the tool out of the catalogue
    # so the model is not offered a question it cannot ask.

    def run(self, sdk, **kwargs):
        """Run ask question."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            return sdk.fail("question is required.")

        answer_type = (kwargs.get("type") or "string").strip().lower()
        if answer_type not in TYPES:
            return sdk.fail(f"type must be one of: {', '.join(sorted(TYPES))}.")

        try:
            timeout = min(max(int(kwargs.get("timeout", DEFAULT_TIMEOUT)), 1),
                          MAX_TIMEOUT)
        except (TypeError, ValueError):
            return sdk.fail("timeout must be an integer number of seconds.")

        choices, enum_error = _clean_enum(kwargs.get("enum"))
        if enum_error:
            return sdk.fail(enum_error)

        try:
            value = sdk.ui.ask(
                question,
                title=(kwargs.get("title") or "Question for you").strip()
                      or "Question for you",
                type=answer_type,
                choices=choices,
                required=kwargs.get("required", True),
                default=kwargs.get("default"),
                timeout=timeout)
        except sdk.Denied:
            # Cancelling is an answer of a kind, and a different one from
            # never replying — so it gets its own message and the model is
            # told to stop rather than to wait.
            return sdk.fail("The user cancelled the question. Do not re-ask; "
                            "continue without the answer or ask what they want instead.")
        except sdk.Failed as failed:
            return sdk.fail(f"The question was not answered: {failed.error}")

        return sdk.ok({"value": value, "type": answer_type},
                      llm_summary=f"User answered: {value!r}")


def _clean_enum(value):
    """Sanitize the caller-supplied choice list; ``(choices, error)``.

    Some models emit option *objects* or empty strings instead of plain values.
    An enum with unanswerable entries would trap the user on a question with no
    valid choice, so extract what we can and reject the call back to the agent
    when nothing usable remains. This stays in the tool rather than moving to
    the handler: it is a workaround for how models write arguments, not a rule
    about what may be asked.
    """
    if value in (None, ""):
        return None, None
    if not isinstance(value, list):
        return None, "enum must be an array of plain non-empty strings."
    cleaned = []
    for item in value:
        if isinstance(item, dict):  # tolerate {"label": ..., "value": ...} shapes
            item = item.get("value") or item.get("label") or ""
        text = str(item).strip()
        if text:
            cleaned.append(text if isinstance(item, str) else item)
    if not cleaned:
        return None, ("enum contained no usable choices (empty strings or empty objects). "
                      "Re-call with each option as a plain non-empty string, e.g. "
                      '["Option A", "Option B"].')
    return cleaned, None
