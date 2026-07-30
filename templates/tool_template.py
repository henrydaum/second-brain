"""
TOOL TEMPLATE
=============
A tool is something the agent can call during a turn. Reference for authoring
one; not imported by the running system.

Read docs/SDK.md first — it covers the Request surface, the return idiom, and what
the validator rejects. Read sandbox/guest/bases.py for every attribute a tool
can declare. This file covers only what is specific to tools and cannot be
guessed from either.

  Where it goes:  DATA_DIR/sandbox_plugins/tools/tool_<name>.py
                  (or plugins/tools/ only when it is true kernel behavior)
  Filename:       must start with "tool_" — discovery is by filename
  Entry point:    run(self, sdk, **kwargs)

Live-loading is automatic when the plugin watcher is enabled: adding, editing,
or deleting the file loads, reloads, or unloads the tool.


THE DESCRIPTION IS THE PROMPT
-----------------------------
`description` is not documentation — it is the text the model reads when
deciding whether to call this tool, and it is the single biggest factor in
whether the tool gets used correctly. Write it as short operational docs:

  - what it does
  - when it is the right choice
  - the most important limit or failure mode

    "Read a UTF-8 text file by exact path. Use when the user has named a
     specific local file. Fails if the path is missing or is not readable text."

Vague hype and trigger words ("ALWAYS use this!") make a tool worse, not
better. Weak models read this literally, so ambiguity here costs real accuracy.


WHAT THE MODEL SEES VERSUS WHAT THE USER SEES
---------------------------------------------
This split is Second Brain specific and easy to get backwards:

  return value / data   -> the frontend, for display. NEVER sent to the model.
  llm_summary           -> what the model receives as the tool result.
  attachments           -> file paths the frontend renders (images, exports).

A bare `return x` sends a rendering of x to the model, which is what you want
most of the time. Reach for `sdk.ok(...)` when the model needs a different
view of the answer than the user does — a thousand rows to display, one
sentence to reason about:

    return sdk.ok(rows, llm_summary=f"{len(rows)} matching rows.")

Put the facts the model needs to act on next in `llm_summary`: what was found,
what changed, what failed, and any counts or paths it will need.


TOOLS VERSUS TASKS
------------------
Tools are called on demand and answer immediately. Tasks run in the background
over every file that appears. If it processes a corpus, it is a task.


BUDGETS AND BACKGROUND SAFETY
-----------------------------
  max_calls = 3          how many times the agent may call this per message
  background_safe = True whether it may run with no human present

Set `background_safe = False` for anything that asks the user something —
sdk.ui.ask, sdk.ui.approve, anything interactive. The kernel refuses such tools
from unattended sessions (scheduled subagents, background drivers) rather than
letting them hang forever waiting for an answer nobody is there to give.

Both are authority-bearing, so both are clamped. Declaring a bigger number is a
request, not a grant.


The three examples below are three separate tools, shown together for
contrast. A real file declares exactly ONE plugin class — the validator
enforces it, because discovery expects one class per file.
"""

from guest.bases import BaseTool


class WordCount(BaseTool):
    """The smallest useful tool: one Request, a plain return."""

    name = "word_count"
    description = (
        "Count the words in a UTF-8 text file. Use when the user asks how "
        "long a specific local file is. Fails if the path is not readable text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Exact path to the file to count.",
            },
        },
        "required": ["path"],
    }

    # Your text in the agent's system prompt, present only while you are
    # installed. Point-of-use guidance belongs here rather than in the kernel's
    # static prompt, which every turn pays for whether or not you exist. A
    # plain string is read from this file without importing it and costs
    # nothing; see RecentNotes below for the shape that depends on live state.
    agent_prompt = (
        "## Counting words\n"
        "word_count reads one UTF-8 file. For many files, glob first."
    )

    def run(self, sdk, path):
        """Read the file and count its words."""
        # No error branch: if the read fails, that failure becomes this tool's
        # failure, carrying the original reason. Which is what the caller
        # wanted anyway.
        return len(sdk.fs.read(path).split())


class RecentNotes(BaseTool):
    """Shows the two things worth demonstrating: the user/model split, and
    reaching user-owned rows through the ``my_`` prefix."""

    name = "recent_notes"
    description = (
        "List the current user's most recently updated conversations. Use to "
        "orient yourself before answering questions about past discussions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "How many to return. Default 10.",
                "default": 10,
            },
        },
    }

    def agent_prompt(self, sdk):
        """The other shape of the same declaration: a method, when the text
        depends on something only the running system knows.

        Same name as the string form on purpose — the collector accepts
        either. This one is a real call into your box, so the kernel caches
        the result until your plugin reloads. Keep it cheap and keep it
        stable, and reach for the string form whenever you can.
        """
        return (f"## Recent notes\n"
                f"Conversations live under {sdk.paths.get('data')}.")

    def run(self, sdk, limit=10):
        """Query the current user's conversations, newest first."""
        # my_conversations is expanded by the kernel to this user's rows only.
        # Querying the base table directly is refused — see docs/SDK.md.
        rows = sdk.db.query(
            "SELECT title, updated_at FROM my_conversations "
            "ORDER BY updated_at DESC LIMIT ?",
            [int(limit)],
        )
        if not rows:
            return sdk.ok([], llm_summary="No conversations yet.")

        # The user gets the table; the model gets the titles it needs to
        # reason about. Sending a hundred rows of SQL to the model would
        # spend context on formatting it cannot use.
        titles = ", ".join(r["title"] for r in rows[:5])
        return sdk.ok(rows, llm_summary=f"{len(rows)} conversations. Most recent: {titles}.")


class FetchPage(BaseTool):
    """Demonstrates the one case where catching a failure is right."""

    name = "fetch_page"
    description = (
        "Fetch a URL and return its body. Use when the user gives a specific "
        "link. Requires the user's approval on each new host."
    )
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
        "required": ["url"],
    }

    # Network access is unsafe by policy, so this asks the user. Nothing to
    # declare for that — the gate decides. `requests` below is advisory only:
    # it lets someone installing this see what it intends to do.
    requests = ["net.http"]

    def run(self, sdk, url):
        """Fetch the page, turning a refusal into something the model can act on."""
        try:
            response = sdk.net.http(url)
        except sdk.Denied:
            # Worth catching: the model should stop retrying and tell the user.
            # A transport failure is NOT worth catching — let it raise.
            return sdk.fail("The user declined the request to fetch that URL.")
        return sdk.ok(response["body"], llm_summary=f"Fetched {url} ({response['status']}).")
