"""Tool for proposing a plan and leaving plan mode after approval."""

from plugins.BaseTool import BaseTool, ToolResult


class ProposePlan(BaseTool):
    """Propose plan."""
    name = "propose_plan"
    description = (
        "Propose a plan for the user to approve. Use this in plan mode after "
        "you have inspected enough context. Approval exits plan mode; denial keeps plan mode active. "
        "The user can also approve one turn with all permission dialogs auto-approved."
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
    auto_register = False

    def run(self, context, **kwargs) -> ToolResult:
        """Run propose plan."""
        title = (kwargs.get("title") or "Proposed plan").strip()
        plan = (kwargs.get("plan") or "").strip()
        if not plan:
            return ToolResult.failed("plan is required.")
        ask = getattr(context, "request_user_input", None)
        if ask is None:
            return ToolResult.failed("Plan approval is not available — no live session is configured.")
        service = (getattr(context, "services", None) or {}).get("plan_mode")
        if service is None:
            return ToolResult.failed("Plan mode service is not available.")
        body = f"{title}\n\n{plan}"
        choices = ["approve", "approve_full_permissions", "deny"]
        req = ask("Approve plan?", body, type="string", enum=choices)
        if not req.wait(timeout=3600.0):
            req.metadata["timed_out"] = True
            if getattr(context, "runtime", None) is not None and getattr(context, "session_key", None):
                context.runtime.answer_request(context.session_key, req.id, False)
            return ToolResult.failed("Plan approval timed out.")
        choice = req.value
        if req.metadata.get("cancelled") or choice == "deny":
            return ToolResult.failed("Plan denied. Stop and ask the user what they would like to do differently. Plan mode is still active.")
        runtime = getattr(context, "runtime", None)
        session_key = getattr(context, "session_key", None)
        if runtime is not None and session_key:
            session = runtime.sessions.get(session_key)
            message = "Plan approved. Plan mode is off, and permission dialogs are auto-approved for this turn." if choice == "approve_full_permissions" and getattr(session, "busy", False) else "Plan approved. Plan mode is off."
            if service is not None:
                service.approve(session_key, full_permissions=choice == "approve_full_permissions" and getattr(session, "busy", False), message=message)
        if choice == "approve_full_permissions":
            enabled = bool(service is not None and runtime is not None and session_key and service.has_full_permissions_this_turn(runtime.sessions.get(session_key)))
            summary = "Plan approved. Plan mode is off, and permission dialogs are auto-approved for this turn." if enabled else "Plan approved. Plan mode is off."
            return ToolResult(data={"approved": True, "full_permissions_this_turn": enabled}, llm_summary=summary)
        return ToolResult(data={"approved": True, "full_permissions_this_turn": False}, llm_summary="Plan approved. Plan mode is off.")
