"""Invoke a registered slash command through the guarded SDK request."""

dependencies_files = []
dependencies_pip = []
requests = ["command.call"]

from guest.bases import BaseTool

class SlashCommand(BaseTool):
    name = "slash_command"
    description = (
        "Invoke a registered slash command in one shot. This uses the user's "
        "command surface and therefore requires kernel approval. Commands with "
        "gated actions receive that approval through the same guarded call."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Command without leading slash."},
            "args": {"type": "object", "description": "Structured command arguments."},
        },
        "required": ["name"],
    }
    requires_services = []

    def run(self, sdk, **kwargs):
        name = str(kwargs.get("name") or "").strip().lstrip("/")
        args = kwargs.get("args") or {}
        if not name:
            return sdk.fail("A command name is required.")
        if not isinstance(args, dict):
            return sdk.fail("'args' must be an object.")
        try:
            output = sdk.commands.run(name, **args)
        except sdk.Denied as error:
            return sdk.fail(
                f"Command '/{name}' was denied: {error}. STOP and do not retry."
            )
        except sdk.Failed as error:
            return sdk.fail(f"Command '/{name}' failed: {error}")
        text = "" if output is None else str(output)
        return sdk.ok(
            {"command": name, "output": output},
            llm_summary=text or f"/{name} ran.",
        )
