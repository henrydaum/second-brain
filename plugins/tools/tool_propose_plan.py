"""Tool for proposing a plan and leaving plan mode after approval."""

from plugins.BaseTool import BaseTool, ToolResult


class ProposePlan(BaseTool):
    """Propose plan."""
    name = "propose_plan"
    description = (
        "Propose a plan for the user to approve. Use this in plan mode after "
        "you have inspected enough context. Approval exits plan mode; denial keeps plan mode active."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the plan."},
            "plan": {"type": "string", "description": "The proposed plan body."},
        },
        "required": ["title", "plan"],
    }
    max_calls = 3
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        """Run propose plan."""
        title = (kwargs.get("title") or "Proposed plan").strip()
        plan = (kwargs.get("plan") or "").strip()
        if not plan:
            return ToolResult.failed("plan is required.")
        ask = getattr(context, "request_user_input", None)
        if ask is None:
            return ToolResult.failed("Plan approval is not available — no live session is configured.")
        body = f"{title}\n\n{plan}"
        req = ask("Approve plan?", body, type="boolean")
        if not req.wait(timeout=3600.0):
            req.metadata["timed_out"] = True
            if getattr(context, "runtime", None) is not None and getattr(context, "session_key", None):
                context.runtime.answer_request(context.session_key, req.id, False)
            return ToolResult.failed("Plan approval timed out.")
        if req.metadata.get("cancelled") or not req.approved:
            return ToolResult.failed("Plan denied. Plan mode remains active.")
        runtime = getattr(context, "runtime", None)
        session_key = getattr(context, "session_key", None)
        if runtime is not None and session_key:
            runtime.set_plan_mode(session_key, False)
        return ToolResult(data={"approved": True}, llm_summary="Plan approved. Plan mode is off.")
