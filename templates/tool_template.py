"""
TOOL TEMPLATE
=============
This file is a self-contained reference for creating new tools.
It is NOT imported by the running system — it exists for LLM consumption only.

Write tools in the same voice the system expects elsewhere:
- grounded and practical
- explicit about what the tool does
- clear about when to use it
- clear about important limits or safety constraints

To create a new tool:
  1. Use build_plugin(plugin_type="tool", file_name="tool_<your_name>.py",
     action="create", code="...") to write the file to the sandbox.
  2. The code MUST inherit from BaseTool and include:
       from plugins.BaseTool import BaseTool, ToolResult
  3. Fill in the class attributes and implement run().
  4. Hot-reload picks it up automatically — no restart needed.
  5. If the tool needs extra packages, install them first with
     run_command(command="pip install <pkg>", justification="...", timeout=300).

build_plugin automatically validates:
  - Correct import (from plugins.BaseTool import BaseTool, ToolResult)
  - Class inheriting BaseTool with a `name` attribute
  - No name collisions with baked-in tools
  - File naming conventions (must start with "tool_")


AUTO-DISCOVERY RULES
--------------------
- File must be in plugins/tools/ (baked-in) or the sandbox tools dir
- File name must start with "tool_"
- Class must inherit from BaseTool
- One tool class per file (recommended)


TOOLS vs TASKS
--------------
Tasks run in the background on every file (batch processing).
Tools are called on-demand and return results immediately.

Tools can be called by:
  - The LLM agent (subject to the active agent_profile's tool mode and tools_list)
  - The user via CLI: /call tool_name {"arg": "value"}
  - Other tools via context.call_tool("tool_name", arg="value")


TRIGGERING EVENT TASKS FROM A TOOL
----------------------------------
Event-triggered tasks (trigger="event") subscribe to bus channels. A tool
can fire one by emitting on a channel the task declared, OR by calling
the controller helper which emits on the task's first channel:

    from events.event_bus import bus
    bus.emit("trigger.cluster_now", {"reason": "user asked"})
    # or, without knowing the channel name:
    # ctrl.trigger_event_task("cluster_embeddings", {"reason": "..."})

This enqueues a run in task_runs; the orchestrator dispatches it on its
next tick. The tool returns immediately — it doesn't wait for the run
to complete. Inspect results in the event-driven section of /tasks.


CONTEXT OBJECT
--------------
Every tool receives a `context` object with:

  context.db        Database instance (SQLite). Key methods:
                      .conn  (raw connection, use with context.db.lock)
                      .lock  (threading.Lock for direct SQL)
                      .get_task_output(table, path) -> list[dict]

  context.config    Global settings dict (from config.json).

  context.services  Dict of {name: service_instance}. Access shared models:
                      llm = context.services.get("llm")
                      embedder = context.services.get("text_embedder")

  context.parse     Parse a file using the Stage 1 parser system:
                      result = context.services.get("parser").parse(path)
                      result = context.services.get("parser").parse(path, "image")

  context.call_tool Call another tool (tool-to-tool chaining):
                      result = context.call_tool("lexical_search", query="hello")
                    Returns a ToolResult.


WRITING GOOD TOOL DESCRIPTIONS
------------------------------
`description` becomes the tool description shown to the LLM.
Write it like short operational documentation:

- say what the tool does
- say when the tool is the right choice
- mention the most important limits or constraints
- avoid vague hype or trigger-word instructions

Good pattern:
  "Search indexed files using both keyword and semantic retrieval. Use this
   when you need the best default search over local files. Optional filters
   can narrow the search."


TOOL RESULT
-----------
Return a ToolResult from run():

  ToolResult(
      success=True,
      data=any_structured_data,    # for frontend display (never sent to LLM)
      llm_summary="text for LLM",  # what the LLM sees as the tool result
      attachment_paths=["path"],  # file paths for frontend attachment rendering
  )

  ToolResult.failed("error message")  # shorthand for failures

`llm_summary` should be concise but informative. Include the facts the model
needs to act on next: what was found, what was changed, what failed, and any
important paths, counts, or constraints.


PARAMETERS (JSON Schema)
-------------------------
The `parameters` dict defines the tool's input arguments using JSON Schema.
This is the exact format used by OpenAI function calling:

  parameters = {
      "type": "object",
      "properties": {
          "query": {
              "type": "string",
              "description": "What to search for in the indexed local files.",
          },
          "top_k": {
              "type": "integer",
              "description": "Max results. Default 5.",
              "default": 5,
          },
      },
      "required": ["query"],
  }

In run(), access these via kwargs: query = kwargs.get("query", "")


CONFIG SETTINGS
---------------
Plugins can declare config settings that appear in frontend config views and are
stored in plugin_config.json. Values are accessible via context.config.get().

  config_settings = [
      ("My Setting Title", "my_setting_key",
       "Description shown in frontend config views.",
       "default_value",
       {"type": "text"}),
  ]

Each entry is a tuple: (title, variable_name, description, default, type_info)

type_info controls the UI widget:
  {"type": "text"}                                          — text field
  {"type": "bool"}                                          — checkbox
  {"type": "json_list"}                                     — JSON array editor
  {"type": "slider", "range": (min, max, divs), "is_float": False} — slider

Multiple plugins can declare the same variable_name — the value is shared.
In run(), access via: context.config.get("my_setting_key", "default")
"""

# =====================================================================
# BASE CLASS (copied from plugins/BaseTool.py for self-containment)
# =====================================================================

import logging
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    error: str = ""
    data: Any = None
    llm_summary: str = ""
    attachment_paths: list[str] = field(default_factory=list)

    @staticmethod
    def failed(error: str) -> "ToolResult":
        return ToolResult(success=False, error=error)


class BaseTool:
    # --- Identity ---
    name: str = ""
    description: str = ""               # doubles as the LLM tool description
    parameters: dict = {}               # JSON Schema for input arguments

    # --- Service requirements ---
    requires_services: list[str] = []   # services that must be loaded

    # --- Agent controls ---
    max_calls: int = 3                  # max times the agent can call this tool per message

    # --- Config settings ---
    config_settings: list = []          # settings shown in frontend config views

    def run(self, context, **kwargs) -> ToolResult:
        """Execute the tool. Return a ToolResult."""
        raise NotImplementedError

    def to_schema(self) -> dict:
        """Export as an OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# =====================================================================
# EXAMPLE: A simple database query tool
# =====================================================================

# from plugins.BaseTool import BaseTool, ToolResult
#
#
# class SQLQuery(BaseTool):
#     name = "sql_query"
#     description = "Execute a read-only SQL query against the local database."
#     parameters = {
#         "type": "object",
#         "properties": {
#             "sql": {
#                 "type": "string",
#                 "description": "A read-only SQL query. Only SELECT and PRAGMA statements are allowed.",
#             },
#         },
#         "required": ["sql"],
#     }
#     requires_services = []
#     max_calls = 5
#
#     def run(self, context, **kwargs):
#         sql = kwargs.get("sql", "").strip()
#         if not sql:
#             return ToolResult.failed("No SQL provided.")
#
#         # Safety: only allow SELECT and PRAGMA
#         first_word = sql.split()[0].upper()
#         if first_word not in ("SELECT", "PRAGMA"):
#             return ToolResult.failed("Only SELECT and PRAGMA allowed.")
#
#         try:
#             with context.db.lock:
#                 cur = context.db.conn.execute(sql)
#                 columns = [d[0] for d in cur.description] if cur.description else []
#                 rows = [list(row) for row in cur.fetchmany(100)]
#             return ToolResult(
#                 data={"columns": columns, "rows": rows},
#                 llm_summary=f"Query returned {len(rows)} row(s).",
#             )
#         except Exception as e:
#             return ToolResult.failed(f"SQL error: {e}")


# =====================================================================
# EXAMPLE: A tool that calls another tool (tool chaining)
# =====================================================================

# from plugins.BaseTool import BaseTool, ToolResult
#
#
# class HybridSearch(BaseTool):
#     name = "hybrid_search"
#     description = (
#         "Search indexed files using both keyword and semantic retrieval. "
#         "Use this when you need the best default search over local files."
#     )
#     parameters = {
#         "type": "object",
#         "properties": {
#             "query": {"type": "string", "description": "What to search for in the indexed local files."},
#             "top_k": {"type": "integer", "description": "Max results.", "default": 5},
#         },
#         "required": ["query"],
#     }
#
#     def run(self, context, **kwargs):
#         query = kwargs.get("query", "")
#         top_k = kwargs.get("top_k", 5)
#
#         # Call sub-tools via context.call_tool
#         lexical = context.call_tool("lexical_search", query=query, top_k=top_k * 10)
#         semantic = context.call_tool("semantic_search", query=query, top_k=top_k * 10)
#
#         # Fuse results...
#         all_results = (lexical.data or []) + (semantic.data or [])
#         return ToolResult(
#             data=all_results[:top_k],
#             llm_summary=f"Found {len(all_results)} results for '{query}'.",
#         )
