"""Run a background agent on a prompt, now — the immediate-run counterpart
to schedule_subagent.

Sandboxed, and almost nothing is left. Opening the child's conversation,
driving its turn, holding the parent's turn open until it reports, and the
deadline that cancels it are all kernel routing now (runtime/subagents.py),
reached through ``sdk.agent``. What remains here is the agent-facing shape:
the parameters, the guidance, and the choice of whether to wait.

The ``task_runs`` LIKE-query that used to locate the running child went with
it — a spawn has a handle now.
"""

dependencies_files = []
dependencies_pip = []
requests = ["agent.spawn"]

from guest.bases import BaseTool


class SpawnSubagent(BaseTool):
    """Spawn subagent."""
    name = "spawn_subagent"
    description = (
        "Spawn a subagent in a separate context window. You will not see the "
        "subagent's intermediate tool calls or outputs, only the final result. "
        "The prompt must be complete and self-contained — the agent cannot ask you "
        "follow-up questions, and it cannot use tools that require user approval. "
        "wait=true (default) blocks and returns the agent's result; wait=false runs "
        "it in the background while you continue, and its completion notice will arrive "
        "in this conversation before your turn ends. Use wait=false to send out multiple "
        "agents at once, useful for deep research and exploring multiple perspectives."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Complete, self-contained instructions for the agent."},
            "title": {"type": "string", "description": "Short title for the agent's conversation."},
            "attachments": {"type": "array", "items": {"type": "string"}, "description": "Optional file paths to attach."},
            "wait": {"type": "boolean", "description": "true (default): block and return the result. false: run in the background and continue."},
            "timeout_seconds": {"type": "integer", "description": "Max seconds the agent may run before it is cancelled. Capped by the subagent_timeout_seconds setting, whose default and maximum are both 600."},
            "profile": {"type": "string", "description": "An agent profile name from the user's config, which narrows the tools the child may use. Omit to give it the same profile you have."},
            "narration": {"type": "string", "description": "A few words on what you are delegating and why, shown to the user beside the call. E.g. 'sending an agent to read the three long PDFs'."},
        },
        "required": ["prompt"],
    }
    requires_services = []
    def run(self, sdk, **kwargs):
        """Run spawn subagent."""
        prompt = (kwargs.get("prompt") or "").strip()
        if not prompt:
            return sdk.fail("No prompt provided.")

        title = (kwargs.get("title") or "Subagent").strip() or "Subagent"
        wait = bool(kwargs.get("wait", True))
        try:
            report = sdk.agent.spawn(
                prompt,
                title=title,
                attachments=[str(p).strip()
                             for p in (kwargs.get("attachments") or [])
                             if str(p).strip()],
                wait=wait,
                timeout_seconds=kwargs.get("timeout_seconds"),
                profile=(kwargs.get("profile") or "").strip() or None,
            )
        except sdk.Failed as refused:
            # Every refusal the kernel makes is worth reporting verbatim: no
            # recursive spawning, the active conversation, a busy child, a
            # missing attachment. Each names what to do differently.
            return sdk.fail(str(refused))

        cid = report.get("conversation_id")
        if not wait:
            return sdk.ok(
                {"conversation_id": cid, "id": report.get("id"),
                 "wait": False},
                llm_summary=(
                    f"Spawned background agent '{title}' in conversation "
                    f"#{cid}. Keep working — its completion notice will be "
                    "delivered to you automatically; do not poll for it."))

        if report.get("state") == "cancelled":
            return sdk.fail(
                f"Agent '{title}' timed out and was cancelled — it produced no "
                f"result (partial transcript in conversation #{cid}). Retry "
                f"with a smaller prompt or a larger timeout_seconds.")
        if not report.get("ok"):
            return sdk.fail(
                f"Agent '{title}' failed: "
                f"{report.get('error') or 'unknown error'} "
                f"(conversation #{cid})")
        return sdk.ok(
            {"conversation_id": cid, "id": report.get("id"), "wait": True},
            llm_summary=((report.get("text") or "").strip()
                         or "(the agent produced no final text)")
                        + f"\n\n(agent '{title}' ran in conversation #{cid})")
