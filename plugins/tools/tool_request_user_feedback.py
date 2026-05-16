"""request_user_feedback: ask the user what they think of the current canvas."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult


class RequestUserFeedback(BaseTool):
    name = "request_user_feedback"
    description = (
        "Ask the user a question about the current canvas — what they want changed, "
        "whether the direction is right, what mood they want next. Use this OFTEN, "
        "especially after every 1-2 compose calls, because you cannot see the canvas "
        "directly — only the user can judge whether it's working. Phrase the question "
        "naturally and offer a concrete suggestion or two so the user has something "
        "to react to."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user. Be conversational and concrete.",
            }
        },
        "required": ["question"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        question = str(kwargs.get("question") or "").strip()
        if not question:
            return ToolResult.failed("`question` is required.")
        return ToolResult(
            data={"question": question},
            llm_summary=f"Asked the user: {question}",
        )
