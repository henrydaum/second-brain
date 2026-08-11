"""
Kernel event channel registry.

Declaring channels in one place is the discipline that keeps the event bus
from becoming a dumping ground. If adding a channel feels like it needs
justification, that's the point — use the bus only when the producer and
consumer are architecturally far apart. For anything tightly coupled or on
the hot path, call the function directly.

**Scope: this file documents channels the *kernel* produces or consumes.**
The bus itself needs no registration (a channel is just a string), so a plugin
owns its own channels — it defines the constant + payload doc at the top of its
module (relative-imported by siblings in the same family) and emits/subscribes
there. A plugin channel must NOT be declared here, even speculatively: the
kernel must not carry contracts for code it doesn't contain (uninstall the
plugin and its channel should vanish with it). Kernel-produced channels may, of
course, be *subscribed to* by plugins — that's the whole point of emitting them
unconditionally.

Payload shapes are documented here, not enforced at runtime.
"""

# ── Active channels ────────────────────────────────────────────────

APPROVAL_REQUESTED = "approval_requested"
"""A conversation session is waiting for user approval or typed input.
Payload: a StateMachineApprovalRequest object."""

APPROVAL_SETTLED = "approval_settled"
"""A question stopped waiting: answered, cancelled, or denied by timeout.

The counterpart to APPROVAL_REQUESTED, and the half that was missing. A
frontend learns a question exists by being handed one to render, and used to
have no way at all to learn it had stopped existing — another frontend can
answer it, the 300s dialog timeout denies it by name, and neither of those
sends anything. A surface that cannot be told has to poll to find out, and a
surface that does not poll shows a dialog that can no longer be answered.

Emitted from the one place every resolution funnels through, so "settled" means
the phase frame is gone rather than "somebody called resolve".
Payload:
    session_key: str
    request_id:  str
    reason:      str — "answered" | "cancelled\""""

FORM_REQUESTED = "form_requested"
"""A restored session is sitting on a suspended command/tool form and needs the
current field re-prompted. Used only on restore: in normal flow the form rides
back as ``RuntimeResult.form`` on the submit() that produced it (tightly
coupled), but after a process restart there is no submit() in flight, so the
producer (runtime restore) and consumer (frontend) are far apart — same reason
APPROVAL_REQUESTED is re-emitted on restore.
Payload:
    session_key: str
    form:        dict — the descriptor render_form_field expects (see
                        runtime/dispatch.py decorate_form)"""

TASK_STARTED = "task_started"
"""A task run was dispatched to a worker — completes the triad with
TASK_COMPLETED / TASK_FAILED for a live pipeline view. Emitted unconditionally
(even with no subscribers) so a plugin can observe the pipeline by subscribing,
never by editing the kernel.
Payload (path-triggered tasks):
    task_name: str
    paths:     list[str]   — the batch dispatched together
Payload (event-triggered tasks):
    task_name: str
    run_id:    str"""

TASK_COMPLETED = "task_completed"
"""A task finished successfully.
Payload (path-triggered tasks):
    task_name:    str
    path:         str
    rows_written: int
    duration_s:   float
Payload (event-triggered tasks):
    task_name:    str
    run_id:       str
    rows_written: int
    duration_s:   float"""

TASK_FAILED = "task_failed"
"""A task failed.
Payload (path-triggered tasks):
    task_name: str
    path:      str
    error:     str
Payload (event-triggered tasks):
    task_name: str
    run_id:    str
    error:     str"""

SERVICE_LOADED = "service_loaded"
"""A service finished (un)loading or was swapped. Lets the orchestrator
re-check tasks that were blocked on services without reaching sideways into it. Emitted on load, unload, and hot-reload.
Payload:
    name:   str   — service name (may be None for bulk events)
    loaded: bool  — True after load, False after unload"""

TOOLS_CHANGED = "tools_changed"
"""A tool was registered, re-registered, or unregistered. Lets frontends
rescope running agents so build_plugin / unload_plugin updates take effect
without /restart.
Payload:
    name:   str — tool name
    action: str — 'registered' or 'unregistered'"""

TASKS_CHANGED = "tasks_changed"
"""A task was registered or unregistered. Task registration creates a new
output table via ensure_output_table, so agents rebuild their prompt context.
Payload:
    name:   str — task name
    action: str — 'registered' or 'unregistered'"""

CHAT_MESSAGE_PUSHED = "chat_message_pushed"
"""Text belonging to the conversation that has no ``RuntimeResult`` to ride on.

Narrower than it once was. This used to be the channel for anything reaching
the user out of band, which meant it carried both halves of a distinction
frontends needed and could not make; the announcement half moved to
NOTIFICATION_PUSHED. What is left is conversation: the model's mid-turn
narration alongside a tool call, and the files a tool shows through
``sdk.ui.render``. Both are the agent's turn speaking, and both belong in the
chat view of every frontend, which is why they are still here.

``kind`` / ``source`` / ``source_id`` are vestigial — ``BaseFrontend`` reads
none of them, and a producer that wants attribution shown wants
NOTIFICATION_PUSHED instead.
Payload:
    message:  str            — the body text to display (required)
    title:    str (optional) — rendered as a header above the message
    kind:     str (optional) — categorical label (e.g. "note", "alert"); if
                               title is empty, may be used as a fallback header
    source:   str (optional) — identifier for the producer; frontends may
                               show this as attribution
    source_id:str (optional) — producer-specific id
    attachments: list[str] (optional) — local file paths to render alongside
                               the text. A push may carry these with no
                               message at all, which is what ``sdk.ui.render``
                               with no caption sends."""

NOTIFICATION_PUSHED = "notification_pushed"
"""Something the user should be *told about*, as distinct from something said
to them in conversation.

The distinction CHAT_MESSAGE_PUSHED could not draw. That channel carries two
populations: text belonging to the agent's own turn — the model's mid-turn
narration, a tool showing a file through ``sdk.ui.render`` — and announcements
from elsewhere in the system that merely have nowhere else to appear. A
frontend receiving only ``render_messages`` cannot tell them apart, so a plugin
registering itself and the agent answering a question arrive as the same kind
of thing.

The line is *who was speaking*: a push made while the agent's turn owns the
session is conversation and stays on CHAT_MESSAGE_PUSHED; everything else is a
notification. That is decided at each emit site rather than inferred from the
channel, so a future producer cannot land on the wrong side by accident.

``source`` is stamped by the kernel, never by the producer that asked. For
sandboxed code it is read off the live provenance chain (see
``runtime.notifications.notify``), which is precisely the part of a chain a box
cannot state about itself — the same property ``sandbox.approval.describe_asker``
and the ledger's ``actor_id`` rely on.

Delivery and origin are different fields, and both are needed. A scheduled
agent's result belongs to a background session nobody is watching; it has to be
*shown* on whatever surface the user is actually looking at.
Payload:
    title:      str            — short header; what the notification is
    body:       str            — the text (required unless title carries it)
    source:     str            — kernel-stamped producer identity
    source_id:  str (optional) — producer-specific id
    level:      str            — 'info' | 'success' | 'warning' | 'error'
    session_key:str (optional) — delivery target; absent means broadcast
    source_session_key: str (optional) — the session it came *from*
    conversation_id: int (optional) — the conversation it is about
    load_hint:  str (optional) — pre-rendered text affordance for reaching
                                 that conversation, for frontends with no
                                 richer way to offer it. Structured clients
                                 should use conversation_id and ignore this.
    notification_id: int (optional) — the persisted row's id, absent when the
                                 notification was not persisted
    sent_at:    float          — epoch seconds"""

AGENT_TEXT_DELTA = "agent_text_delta"
"""A fragment of streamed assistant text (emitted only when the session's
frontend declares ``FrontendCapabilities.supports_streaming`` and the active
LLM backend supports streaming — both halves have to be able to). Frontends
that declare it render deltas incrementally via ``render_stream_delta``;
for everyone else nothing is emitted at all and the same text arrives as
whole messages.
Payload:
    session_key: str
    stream_id:   str  — unique per LLM call
    seq:         int  — monotonically increasing per stream
    delta:       str  — raw text fragment ("" on done events)
    done:        bool — stream finished
    aborted:     bool — done-only: stream ended without a usable final
                        (error / cancel / compaction retry)
    final_text:  str (optional) — clean done only: the CLEANED full text,
                        byte-identical to what arrives via the whole-message
                        path (RuntimeResult / CHAT_MESSAGE_PUSHED) — the
                        dedup key for frontends that streamed it
    kind:        str (optional) — clean done only: "final" | "narration" """

TOOL_CALL_STARTED = "tool_call_started"
"""The agent started a tool call.
Payload:
    session_key: str
    call_id:     str
    tool_name:   str
    args:        dict — the model's verbatim call, narration included
    narration:   str — the declared "narration" argument, collapsed and capped
                 by ``runtime_config.tool_blurb``; "" when not declared"""

TOOL_CALL_FINISHED = "tool_call_finished"
"""The agent finished a tool call.
Payload:
    session_key: str
    call_id:     str
    tool_name:   str
    ok:          bool
    error:       str (optional)
    narration:   str — byte-identical to the started event's, because a
                 frontend that overwrites its status line in place no longer
                 has the started payload to read it from
    summary:     str — what the call amounted to: the tool's ``llm_summary``,
                 or its ``data`` as JSON when only that was filled in, capped
                 exactly as the transcript row is so the two cannot disagree.
                 "" on failure (``error`` is the outcome then) and "" when the
                 tool reported neither half"""

COMMAND_CALL_STARTED = "command_call_started"
"""The runtime started a slash command.
Payload:
    session_key:  str
    call_id:      str
    command_name: str
    args:         dict"""

COMMAND_CALL_PROGRESSED = "command_call_progressed"
"""The runtime collected another slash-command form value.
Payload:
    session_key:  str
    call_id:      str
    command_name: str
    args:         dict"""

COMMAND_CALL_FINISHED = "command_call_finished"
"""The runtime finished a slash command.
Payload:
    session_key:  str
    call_id:      str
    command_name: str
    ok:           bool
    error:        str (optional)"""


# ── Conversation lifecycle ─────────────────────────────────────────
# Plugins (tools, tasks, services) subscribe to these to react to what
# is happening inside the state machine without having to reach into
# ConversationRuntime directly. Frontends emit and consume them too.

SESSION_CREATED = "session_created"
"""A new RuntimeSession was created (or replaced via /new or load_history).
Payload:
    session_key: str
    agent_profile: str"""

SESSION_CLOSED = "session_closed"
"""A RuntimeSession was discarded (replaced, deleted, app shutdown).
Payload:
    session_key: str"""

SESSION_PHASE_CHANGED = "session_phase_changed"
"""The session's phase transitioned (awaiting_input -> calling_tool, etc.).
Payload:
    session_key: str
    old_phase:   str
    new_phase:   str"""

SESSION_TURN_CHANGED = "session_turn_changed"
"""Turn priority moved between participants on a session.
Payload:
    session_key: str
    from_actor:  str
    to_actor:    str"""

SESSION_MESSAGE = "session_message"
"""One transcript row landed on a session — a complete live feed of the
conversation. Agent-turn rows (assistant text, assistant tool-call rows, tool
results, drained mid-turn user messages) are emitted by the loop's single
record point (``ConversationLoop._record``); user-side rows by the dispatch
layer. Subscribers building per-message consumers (live transcript views,
memory extractors) get every row in order without polling the DB.
Payload:
    session_key:  str
    role:         str   — "user" | "assistant" | "tool"
    content:      str
    actor_id:     str   — "user" | "agent"
    attachments:  list[dict]            — user rows: the files that message
                                          carried, as {path, file_name,
                                          modality, extension}. Empty for
                                          almost every row; ``content`` is
                                          what the person typed and nothing
                                          else, so this is the only thing
                                          saying a file arrived.
    name:         str (optional)        — tool rows: the tool name
    tool_call_id: str (optional)        — tool rows: id pairing with the call
    tool_calls:   list[dict] (optional) — assistant rows that request tools"""

SESSION_TURN_STARTED = "session_turn_started"
"""An agent turn is about to be driven (foreground and background alike).
Pairs with SESSION_TURN_COMPLETED — including on crash, which completes with
``ok: False`` — so live surfaces can show busy state without watching flags.
Payload:
    session_key:     str
    conversation_id: int | None
    actor_id:        str — "agent" """

SESSION_TURN_COMPLETED = "session_turn_completed"
"""One driven agent turn finished. Emitted per drive from the runtime's
single drive site (interim drives of a restarted turn — e.g. escalation —
do not emit; the re-driven turn's completion covers the logical turn).
Handlers run synchronously on the drive thread — heavy consumers (memory
extraction, skill reflection) should be event-triggered pipeline tasks
(``trigger="event"``), which just enqueue a task_runs row here and do the
work on the orchestrator's schedule.
Payload:
    session_key:     str
    conversation_id: int | None
    user_id:         int — owner of the session (scope memory/skills per user)
    ok:              bool — False when the drive crashed (error present)
    cancelled:       bool (ok drives only) — the turn was interrupted
    error:           str (crash only)
    final_text:      str
    new_messages:    list[dict]
    attachments:     list[str]"""

SESSION_COMPACTED = "session_compacted"
"""The loop compacted a session's history into a summary. The summary text
rides along so subscribers (memory builders, live UIs showing a "condensed"
marker) don't have to re-read the compaction marker table.
Payload:
    session_key:        str
    conversation_id:    int | None
    messages_compacted: int — history rows replaced by the summary
    summary:            str"""

AGENT_LLM_CALL_STARTED = "agent_llm_call_started"
"""The loop is issuing one LLM request (there are several per agent turn when
tools are involved). Lets frontends show a "thinking" indicator even when
streaming is off, and lets observers meter model usage per session.
Payload:
    session_key: str
    model:       str | None
    streaming:   bool"""

AGENT_LLM_CALL_FINISHED = "agent_llm_call_finished"
"""The LLM request finished (pairs with AGENT_LLM_CALL_STARTED).
Payload:
    session_key:    str
    model:          str | None
    ok:             bool
    error:          str | None
    duration_s:     float
    prompt_tokens:  int | None
    has_tool_calls: bool"""

SESSION_AGENT_PROFILE_CHANGED = "session_agent_profile_changed"
"""A plugin or command changed the agent profile pinned to a session.
Payload:
    session_key:  str
    old_profile:  str
    new_profile:  str"""

SESSION_SECURITY_MODE_CHANGED = "session_security_mode_changed"
"""The security mode a conversation answers approval dialogs with changed
(``/mode``, or an approval answered "for the rest of this turn"). Frontends
that show a persistent posture indicator subscribe to this; so does anything
wanting to notice that a session stopped asking.
Payload:
    session_key:     str
    conversation_id: int | None
    mode:            str   (lockdown | ask | yolo)
    scope:           str   (conversation | turn)"""

SYSTEM_PROMPT_EXTRA_CHANGED = "system_prompt_extra_changed"
"""A plugin added/updated/removed a system prompt extra on a session.
Useful for frontends or subscribers that want to surface what's pinned to
the agent's prompt.
Payload:
    session_key: str
    key:         str
    value:       str | None  (None on removal)"""

SESSION_CONVERSATION_CHANGED = "session_conversation_changed"
"""A live session switched to (or created) a conversation, or the one it is
showing was retitled. Frontends with a persistent surface (pinned banner,
window title, sidebar highlight) subscribe to mirror "where am I?" without
polling. Emitted unconditionally.
Payload:
    session_key:     str
    conversation_id: int
    title:           str"""

SESSION_CONVERSATION_ENDED = "session_conversation_ended"
"""A live session stopped being in a conversation — the other half of
SESSION_CONVERSATION_CHANGED, and the one that says which conversation the
work just finished in.

CHANGED names only the conversation being switched *to*, because its
subscribers are frontends redrawing "where am I?". Anything reasoning about a
conversation as a *unit of work* needs the opposite: the id being left behind.
Reflection, summarization and memory extraction all want this one — a
conversation going quiet is a self-contained episode in a way that a calendar
day is not.

Emitted when the session switches away (``/new``, ``/clear``, loading another
conversation), when the session closes, and when the conversation is deleted
out from under it. A crash emits nothing, so a consumer that must not lose work
needs its own idempotent record of what it has already handled rather than
treating this as exactly-once.

**Subagents end conversations too, and this is the trap.** A spawned child gets
its own conversation and closes its session when it finishes
(``runtime/subagents.py`` ``_run``), so every subagent completion emits here.
A consumer that *spawns* an agent in response to this channel therefore feeds
itself: the child it started closes, that emit arrives, and the consumer reacts
to its own work. Filter on ``session_key`` — a child's key starts with
``spawn_subagent:`` (``subagents.is_subagent_session``) — and on the
conversation's ``Subagent`` category for anything the sweep-style backstop
might pick up later. The channel reports every ending on purpose rather than
hiding the children, because "which endings count" is the consumer's question,
not the kernel's.

Payload:
    session_key:     str — ``spawn_subagent:<cid>`` for a child
    conversation_id: int — the one being left
    user_id:         int
    reason:          str — 'switched' | 'closed' | 'deleted'"""

CONVERSATION_CHANGED = "conversation_changed"
"""The conversation *catalog* changed, as opposed to a live session (SESSION_*).
Lets a frontend refresh a conversation list/sidebar without polling. The kernel
emits created/deleted/recategorized; a retitling plugin (e.g. update_titles)
emits its own 'retitled'. Emitted unconditionally so a plugin can subscribe
without kernel edits.
Payload:
    action:          str — 'created' | 'deleted' | 'recategorized' | 'retitled'
    conversation_id: int
    user_id:         int (optional) — owner, when known
    category:        str | None (optional) — for created / recategorized"""


# ── Subagents ──────────────────────────────────────────────────────

SUBAGENT_SPAWN = "subagent.spawn"
"""Run an agent on a prompt, in its own conversation, in the background.

The kernel subscribes to this itself (runtime/subagents.py) — it is how a
*scheduled* spawn arrives, since a Timekeeper job can only fire a channel. An
immediate spawn does not come through here at all; it goes straight to the
registry via the ``agent.spawn`` Request.

The string is what it was when spawning lived in a store task, so jobs created
before the kernel absorbed it keep firing.
Payload:
    prompt:                 str — required; complete and self-contained
    title:                  str (optional) — names the child's conversation
    conversation_id:        int (optional) — reuse this conversation; a
                            recurring job has its own written back after the
                            first run, so it accumulates one transcript
    attachments:            list[str] (optional) — file paths
    report_session_key:     str (optional) — session whose *agent-facing*
                            message queue receives the child's report. Unset
                            for scheduled jobs, whose delivery surface is the
                            user-facing push instead.
    report_conversation_id: int (optional) — drops the report if that session
                            has since moved to a different conversation"""


# ── Configuration ──────────────────────────────────────────────────

CONFIG_CHANGED = "config_changed"
"""Persisted configuration was written. Lets multi-client frontends resync a
settings panel without polling. Emitted unconditionally so a plugin can
subscribe without kernel edits.
Payload:
    scope: str — 'core' | 'plugin'  (user-scoped settings live in the users
                 table, written elsewhere, and are not covered here)
    keys:  list[str] — the setting *names* that changed, sorted. Names only,
                 never values: config holds tokens, and this is the same rule
                 the ledger's config_save row follows. Empty when a save
                 rewrote the file without changing anything."""


# ── Reserved (kernel-owned, not yet emitted) ───────────────────────
# Future *kernel* channels — the producer would live in the kernel but doesn't
# exist yet. Documented so the work has an obvious home instead of an ad-hoc
# name. (Plugin-owned futures are not listed here; they belong to the plugin.)
#
# TABLE_WRITTEN        — after DB.write_outputs; finer-grained than TASK_COMPLETED
#                        (which already carries rows_written), for reactive
#                        aggregate tasks that key off a specific table
# TOOL_CALL_PROGRESSED — symmetry with COMMAND_CALL_PROGRESSED, once tools can
#                        report incremental progress for a long-running call
