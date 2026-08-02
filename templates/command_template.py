"""
COMMAND TEMPLATE
================
A command is a slash command a person types. Reference for authoring one; not
imported by the running system.

Read docs/SDK.md for the Request surface and sandbox/guest/bases.py for every
attribute a command can declare. This file covers what is specific to commands.

Before writing: read docs/SDK.md, then this entire template. For details not
defined here, inspect sandbox/guest/bases.py (BaseCommand declarations),
sandbox/guest/forms.py (sandbox form values), plugins/native/command.py (host
adapter and approval declarations), and state_machine/forms.py (form
progression). Validate the finished file before registering it.

  Where it goes:  DATA_DIR/workspace/commands/command_<name>.py
  Filename:       must start with "command_"
  Entry points:   run(self, sdk, args) and optionally form(self, sdk, args)

WATCH THE ARGUMENT ORDER: run(self, sdk, args), sdk first.

Commands are for interactive UI and workflow control. If the agent should be
able to call it, write a tool instead.


FORM STEPS ARE PLAIN DICTS
--------------------------
This is the one thing that will catch you. Natively, `form()` returned FormStep
objects. A FormStep is a live kernel object, so sandboxed code cannot hold one
— you return plain dicts and the kernel rebuilds them:

    return [{"name": "text", "prompt": "Enter the note text.", "required": True}]

Recognized keys: name, prompt, required, type, enum, enum_labels, default,
prompt_when_missing, columns. Unknown keys are dropped silently, so a typo
costs you that field rather than raising — check your spelling. `validator`
cannot be passed at all, because it is a callable; validate inside `run`.

Types are coerced before `run` sees them: "string" (default), "integer",
"number", "boolean".

Write each prompt as a user-facing instruction, not a field label:
"Enter the note text." beats "Text".


FORMS SUSPEND, AND THEY SURVIVE RESTARTS
----------------------------------------
Returning steps does not block. The command suspends onto the cache stack and
resumes when the user has answered — across an app restart, if it takes that
long. So `form()` may be called several times for one invocation, and it must
be cheap and free of side effects. Do the work in `run`.

For a dynamic form, look at what has been collected already and return only
the next steps needed. Returning [] means "ready, run now".


RETURN MARKDOWN
---------------
Command output is a string of GitHub-flavored markdown. Each frontend renders
it by its own policy — the REPL aligns tables and strips fences, rich
frontends render natively. Do not invent a structured return type; markdown is
deliberately the interchange format.

Build it with the SDK helpers so it renders consistently everywhere:

    sdk.md.table(headers, rows)     data tables
    sdk.md.card(title, pairs)       describe-style key/value cards

Tables must start their own block — put a blank line before one, or markdown
parsers fold it into the preceding paragraph.

Return None for no visible message.


The two examples below are separate commands, shown together for contrast. A
real file declares exactly ONE plugin class.
"""

from guest.bases import BaseCommand


class Note(BaseCommand):
    """The simplest useful command: one field, one answer."""

    name = "note"
    description = "Append a short note to the current conversation."
    # Guidance added to the agent's system prompt while this command is in
    # scope. A method (``def agent_prompt(self, sdk)``) works too when the
    # text depends on live state.
    agent_prompt = "## Notes\nUse /note for a one-liner worth keeping."
    category = "Conversation"

    def form(self, sdk, args):
        """Ask for the text if it was not given on the command line."""
        if args.get("text"):
            return []          # /note "already provided" — run immediately
        return [{"name": "text", "prompt": "Enter the note text.", "required": True}]

    def run(self, sdk, args):
        """Store the note and confirm."""
        text = (args.get("text") or "").strip()
        if not text:
            return "Nothing to note."
        sdk.conv.append(sdk.session.get()["conversation_id"], "user", f"Note: {text}")
        return f"Noted: {text}"


class Digest(BaseCommand):
    """A dynamic form plus a markdown table — the two things worth copying."""

    name = "digest"
    description = "Summarize recent conversations."
    category = "Conversation"

    def form(self, sdk, args):
        """Build the form one decision at a time."""
        # Step one: what scope? Nothing else can be asked until this is known.
        if "scope" not in args:
            return [{
                "name": "scope",
                "prompt": "Summarize which conversations?",
                "required": True,
                "enum": ["recent", "category"],
                "enum_labels": ["Most recent", "By category"],
            }]
        # Step two depends on the answer to step one.
        if args["scope"] == "category" and "category" not in args:
            categories = sorted({c["category"] for c in sdk.conv.list() if c.get("category")})
            return [{
                "name": "category",
                "prompt": "Which category?",
                "required": True,
                "enum": categories,
            }]
        return []

    def run(self, sdk, args):
        """Render the matching conversations as a table."""
        rows = [c for c in sdk.conv.list()
                if args["scope"] == "recent" or c.get("category") == args.get("category")]
        if not rows:
            return "No matching conversations."

        table = sdk.md.table(
            ["Title", "Updated"],
            [[c["title"], c["updated_at"]] for c in rows[:20]],
        )
        # Blank line before the table, or it folds into the sentence above it.
        return f"**{len(rows)} conversation(s).**\n\n{table}"
