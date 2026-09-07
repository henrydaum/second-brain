# Conversation concurrency audit

Reviewed September 7, 2026, following the desktop/phone incident.

Accepted scope: retain exclusive ownership and prevent concurrent sessions from
disrupting active work. Shared multi-device control is deferred; the architecture
notes below describe a possible future extension, not outstanding patch work.

## Findings and fixes

The earlier loading fix remains necessary: a live conversation must not be
rehydrated from its crash-recovery marker. Reopening its owning session returns
the same object; a different session cannot acquire that conversation.

The deeper audit found additional problems beyond loading:

| Path | Problem | Change |
| --- | --- | --- |
| Action admission | The busy check ran outside the dispatch lock. Startup hooks ran before `busy` was set. | Guards and dispatch now share the session lock, and an ephemeral driver token reserves the turn before hooks run. |
| Completion and restarts | `busy` became false before finish hooks and completion handling finished. A concurrent request could treat unfinished cleanup as idle. | Ownership spans those boundaries and queued follow-up dispatch. Only the owning invocation may release its token. |
| Approval cancellation | Cancel dismissed the approval while its agent driver could continue. | Cancellation is flagged before waking the pending request; pending approval frames are dismissed and work is interrupted. |
| Late approval requests | A tool reaching its approval doorway after cancellation could create another waiting dialog. | Requests arriving after cancellation receive an already-resolved denial. |
| Stale dialog controls | A cancellation carrying an old request ID could affect a newer prompt or turn. | Supplied request IDs must match the current approval. Concurrent answers are serialized. |
| Lifecycle changes | Reset/close checked only `busy`; identity reassignment could ignore a refused close. Delete and clear could mutate a running conversation from another session. | Lifecycle operations atomically check bindings and turn ownership. They also refuse another thread's active command dispatch. Identity is preserved on refusal. |
| Clearing another session's conversation | Clearing refreshed only the requesting session, leaving the actual owner with stale history. | Clear refreshes the session that actually owns the conversation. |
| Late persistence | A superseded session could write its marker. A queued background submission could rewrite history during another turn. | Marker writes require the current session object; queued background submissions do not rewrite history, and completed writes recheck ownership and idleness. |
| Recovery and actor selection | A legacy marker could say idle while leaving priority with the agent. Omitted frontend actor IDs inherited current priority. | An orphaned base-phase agent marker recovers to the user. Frontend dispatch defaults explicitly to the user. |

The public busy snapshot and native frontend routing include the driver
reservation. The narrower `is_turn_in_flight` permission-grant API retains its
existing lifetime so completion callbacks cannot install a turn grant after
the turn's permission cleanup.

The regressions in `tests/test_session_concurrency.py` use thread events to
hold the exact startup, approval and completion boundaries. They test queueing,
cancellation, duplicate answers, stale requests, failed startup, identity and
lifecycle protection, history preservation, and legacy recovery. Earlier
loading and simultaneous-binding tests remain in `tests/test_session_restore.py`.

Validation: the full default suite reported 2,846 passed, 5 skipped, and one
unrelated failure in
`test_store_memory_bundle.py::test_the_corpus_is_actions_and_facts_live_elsewhere`.
That test checks wording in the store memory plugin; running the unchanged HEAD
test against the same store source reproduced its failure independently.

## Two devices on one conversation

The desired design is **multiple authenticated clients attached to one live
conversation controller, with one agent driver**. Both clients should receive
the same events and may submit input, answer the current approval, or cancel
the current turn. Concurrent messages enter one ordered queue; they do not
start independent model loops against the same history.

The current objects do not separate those responsibilities. `RuntimeSession`
owns both conversation state (history, driver, approvals, cancellation) and
transport identity (frontend, user binding, attendance, session key). Aliasing
one instance under two keys would make those identities and event routes
ambiguous. Loading two independent instances would restore the original race.

The HTTP store frontend has another explicit constraint: `_streams` holds one
connection per session key. A second SSE connection replaces the first, as
documented in `HTTP_PROTOCOL.md`. Sharing a thread key alone therefore does not
provide simultaneous event delivery to two devices. This audit did not change
the store frontend or client application.

A complete implementation should:

1. Separate client attachments from the conversation controller; authorize each
   attachment without changing the executing conversation's identity or policy.
2. Broadcast ordered conversation events to all attached clients, with separate
   replay cursors and attendance maintained until the last client disconnects.
3. Supply a reconnect snapshot containing current turn ID, phase, pending
   approval ID and event position. Old replayed dialogs must not become active.
4. Route approvals and cancellation to the controller. Bind controls to request
   IDs and turn IDs, and deduplicate retried client submissions.
5. Test two real clients through the HTTP boundary: simultaneous sends,
   conflicting approval answers, reconnect during approval, one client leaving,
   cancellation during reconnect, and separate-user access refusal.

Until that work is implemented together, retain exclusive runtime ownership.
Reading persisted history does not require creating another mutable session.

## Limits of this patch

These guards coordinate one `ConversationRuntime` in one process. Multiple
kernel processes using the same database would need a database-backed ownership
lease and a generation check on writes. Direct database mutation APIs also
bypass runtime lifecycle guards and should not be used to edit a live transcript.

Cancellation flags stop the conversation loop and invoke available interruption
callbacks. An external operation that already completed cannot be undone, and a
native callable without interruption support must return before its thread ends.
The session remains owned during that interval.

The deterministic tests reproduce code-level races; they do not replay the
original phone incident or constitute an end-to-end two-device test.
