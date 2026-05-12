"""
COMMAND TEMPLATE
================
This file is a self-contained reference for creating slash commands.
It is NOT imported by the running system — it exists for LLM consumption only.

Commands are user-facing conversation actions. They are invoked with `/name`,
can collect form fields, and return text to the frontend. Use commands for
interactive UI/workflow control; use tools for agent-callable capabilities.

Command authoring flow:
  1. Read this template, then read one similar built-in command for style.
  2. Create sandbox_commands/command_<your_name>.py with edit_file.
  3. The code MUST inherit from BaseCommand and include:
       from plugins.BaseCommand import BaseCommand
       from state_machine.conversation import FormStep
  4. Fill in name, description, category, optional form(), and run().
  5. Call test_plugin(plugin_path="sandbox_commands/command_<your_name>.py").
  6. If testing fails, read the error, edit the same file, and retry.
  7. Valid plugins are discovered on startup; plugin_watcher live-loads adds/edits when enabled.
  8. To update: edit the file; plugin_watcher reloads it when enabled.
  9. To remove live and durably: delete the sandbox file; plugin_watcher unloads it when enabled.

AUTO-DISCOVERY RULES
--------------------
- File must be in plugins/commands/ (baked-in) or the sandbox commands dir
- File name must start with "command_"
- Class must inherit from BaseCommand
- Class must have a non-empty `name`

FORMS
-----
`form(args, context)` returns FormStep objects for missing input. The runtime
collects each field, coerces types, and calls run(args, context) when complete.
For dynamic forms, inspect already-collected args and return the next needed
steps. Write each FormStep prompt as a user-facing instruction, not just a
field label: "Enter the note text." is better than "Text".

COMMAND RESULT
--------------
Return a short string for the frontend, or None for no visible message.
Commands should not call the LLM directly; route agent work through tools,
tasks, or runtime methods exposed in context.
"""

# =====================================================================
# BASE CLASS (copied from plugins/BaseCommand.py for self-containment)
# =====================================================================

from state_machine.conversation import FormStep


class BaseCommand:
    """Base command."""
    name: str = ""
    description: str = ""
    category: str = "Other"
    hide_from_help: bool = False
    require_approval: bool = False
    approval_actor_id: str | None = None
    config_settings: list = []

    def form(self, args: dict, context) -> list[FormStep]:
        """Handle form."""
        return []

    def arg_completions(self, context) -> list[str]:
        """Handle arg completions."""
        return []

    def run(self, args: dict, context) -> str | None:
        """Execute `/template` for the active session."""
        raise NotImplementedError


# =====================================================================
# EXAMPLE: A command with a one-field form
# =====================================================================

# from plugins.BaseCommand import BaseCommand
# from state_machine.conversation import FormStep
#
#
# class NoteCommand(BaseCommand):
#     name = "note"
#     description = "Append a short note to memory.md"
#     category = "Memory"
#
#     def form(self, args, context):
#         return [FormStep("text", "Enter the note text to append.", True)]
#
#     def run(self, args, context):
#         text = (args.get("text") or "").strip()
#         if not text:
#             return "No note provided."
#         result = context.tool_registry.call("update_memory", search_block="", replace_block=f"- {text}\n")
#         return result.llm_summary if result.success else result.error


# =====================================================================
# EXAMPLE: A dynamic command form
# =====================================================================

# from plugins.BaseCommand import BaseCommand
# from state_machine.conversation import FormStep
#
#
# class DemoCommand(BaseCommand):
#     name = "demo"
#     description = "Demonstrate dynamic command forms"
#     category = "System"
#
#     def form(self, args, context):
#         steps = [FormStep("mode", "Choose what the demo command should do.", True, enum=["say", "count"])]
#         if args.get("mode") == "say":
#             steps.append(FormStep("text", "Enter the text to return.", True))
#         if args.get("mode") == "count":
#             steps.append(FormStep("n", "Enter the number to count up to.", True, type="integer"))
#         return steps
#
#     def run(self, args, context):
#         if args["mode"] == "say":
#             return args["text"]
#         return ", ".join(str(i) for i in range(1, args["n"] + 1))
