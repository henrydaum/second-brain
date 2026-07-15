"""Subagents runtime extension — the end-of-turn barrier for spawn_agent.

When a session has background children pending (spawn_agent wait=false, or a
wait that timed out), the turn finalizer holds the turn open until each child
finishes or its deadline passes. Completion notices land in the session's
message queue (written by task_spawn_subagent), and the turn is re-driven so
the model sees results before the logical turn ends — never one turn too late.
"""

from __future__ import annotations

dependencies_files = ['tasks/task_spawn_subagent.py']
dependencies_pip = []

import time

from plugins.BaseService import BaseService, EXTENSION

POLL_SECONDS = 1.0
TERMINAL_STATUSES = {"DONE", "FAILED"}


def _run_status(db, cid) -> str | None:
    """Status of the child's task_runs row (payload carries the cid).

    Mirrors tool_spawn_agent.find_run: the tool emits conversation_id as the
    payload's first key, so the serialized match is unambiguous.
    """
    like = f'%"conversation_id": {int(cid)},%'
    try:
        with db.lock:
            row = db.conn.execute(
                "SELECT status FROM task_runs "
                "WHERE task_name = 'spawn_subagent' AND payload_json LIKE ? "
                "ORDER BY created_at DESC LIMIT 1", (like,)).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def _cancel_children(runtime, cids) -> None:
    """Best-effort cancellation of pending child sessions."""
    sessions = getattr(runtime, "sessions", {}) or {}
    for cid in cids:
        event = getattr(sessions.get(f"spawn_subagent:{cid}"), "cancel_event", None)
        if event is not None:
            event.set()


class SubagentsService(BaseService):
    """Registers the end-of-turn barrier for pending background subagents."""

    model_name = "Subagents"
    shared = True
    lifecycle = EXTENSION

    def __init__(self, _config=None):
        super().__init__()
        self.runtime = None
        self._registered = False

    def bind_runtime(self, *, runtime=None, **_):
        """Receive runtime binding and register hooks if already loaded."""
        self.runtime = runtime
        if self.loaded:
            self._register()

    def _load(self) -> bool:
        """Load the extension and register hooks when runtime is available."""
        self.loaded = True
        self._register()
        return True

    def unload(self):
        """Remove hooks."""
        hooks = getattr(self.runtime, "hooks", None) if self.runtime else None
        if hooks is not None:
            hooks.remove(self._barrier)
        self._registered = False
        self.loaded = False

    def _register(self):
        hooks = getattr(self.runtime, "hooks", None) if self.runtime else None
        if hooks is None or self._registered:
            return
        hooks.add_turn_finalizer(self._barrier)
        self._registered = True

    def _barrier(self, session) -> None:
        """Hold the ending turn until pending children finish or time out.

        Finalizers fire after every drive, including restarted ones — the
        pending map empties on the first pass, so re-entry is a no-op.
        """
        pending = getattr(session, "pending_subagents", None)
        if not pending:
            return
        runtime = self.runtime
        db = getattr(runtime, "db", None) if runtime else None
        if db is None:
            session.pending_subagents = {}
            return

        cancel_event = getattr(session, "cancel_event", None)
        delivered = 0
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                _cancel_children(runtime, list(pending))
                session.pending_subagents = {}
                return
            now = time.time()
            for cid in list(pending):
                status = _run_status(db, cid)
                if status in TERMINAL_STATUSES:
                    # The task queued the completion notice before the run
                    # flipped terminal; the re-driven turn's drain absorbs it.
                    pending.pop(cid, None)
                    delivered += 1
                elif pending.get(cid, 0) <= now:
                    # Deadline passed — stop holding the turn; the child keeps
                    # running and its notice arrives at the next turn's drain.
                    pending.pop(cid, None)
            if not pending:
                break
            time.sleep(POLL_SECONDS)

        if delivered and getattr(session, "pending_user_messages", None):
            session.restart_turn = True


def build_services(config) -> dict:
    """Build the subagents service."""
    return {"subagents": SubagentsService(config)}
