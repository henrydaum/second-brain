"""The "sanitizer" layer: kernel invariants checked after every operation.

A fuzzer is only as useful as its oracle. Linux fuzzing leans on KASAN/KMSAN to
turn latent corruption into a loud crash; this module is the analog for the
agent kernel. :func:`check_invariants` inspects a live (headless) kernel and
returns a list of :class:`Violation` describing anything that should never be
true. The fuzzer calls it after each step; an empty list means "still healthy".

The invariants encode the kernel's load-bearing promises:

- **DB integrity** — SQLite ``integrity_check`` / ``foreign_key_check`` clean.
- **Referential sanity** — every conversation is owned by a real user; every
  message belongs to a real conversation.
- **The user dimension** — no live session is bound to a conversation its
  effective user does not own (the one security property the kernel enforces
  on every path; a fuzzer must never be able to violate it).
- **State-machine hygiene** — an idle session (turn handed back to the user,
  no pending approval) has a fully unwound phase-frame stack; ``turn_priority``
  always names a real participant.
- **Registry / thread hygiene** — visible tools are a subset of registered
  tools; background threads don't leak across operations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from pipeline.database import DEFAULT_USER_ID
from state_machine.conversation_phases import FORM_PHASES


@dataclass(frozen=True)
class Violation:
    """One broken invariant. ``check`` is a stable category; ``detail`` is context."""
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.detail}"


def thread_names() -> set[str]:
    """Snapshot of currently-alive thread names (for leak comparison)."""
    return {t.name for t in threading.enumerate()}


def check_invariants(kernel_or_runtime, *, baseline_threads: set[str] | None = None) -> list[Violation]:
    """Return every currently-broken invariant (empty list == healthy)."""
    runtime = getattr(kernel_or_runtime, "runtime", kernel_or_runtime)
    db = getattr(runtime, "db", None) or getattr(kernel_or_runtime, "db", None)
    out: list[Violation] = []

    _check_db(db, out)
    _check_referential(db, out)
    _check_sessions(runtime, db, out)
    _check_registry(runtime, out)
    _check_ledger(db, out)
    _check_threads(baseline_threads, out)
    return out


# ── Database ──────────────────────────────────────────────────────────

def _query(db, sql: str):
    with db.lock:
        return db.conn.execute(sql).fetchall()


def _check_db(db, out: list[Violation]) -> None:
    if db is None:
        return
    try:
        rows = _query(db, "PRAGMA integrity_check")
        result = rows[0][0] if rows else "no result"
        if str(result).lower() != "ok":
            out.append(Violation("db.integrity", str(result)))
    except Exception as e:
        out.append(Violation("db.integrity", f"integrity_check raised: {e!r}"))
    try:
        fk = _query(db, "PRAGMA foreign_key_check")
        if fk:
            out.append(Violation("db.foreign_keys", f"{len(fk)} dangling FK row(s): {fk[:3]}"))
    except Exception as e:
        out.append(Violation("db.foreign_keys", f"foreign_key_check raised: {e!r}"))


def _check_referential(db, out: list[Violation]) -> None:
    if db is None:
        return
    try:
        orphan_convs = _query(
            db,
            "SELECT c.id, c.user_id FROM conversations c "
            "LEFT JOIN users u ON u.id = c.user_id WHERE u.id IS NULL",
        )
        for row in orphan_convs:
            out.append(Violation("ownership.user_missing",
                                  f"conversation {row[0]} owned by absent user {row[1]}"))
    except Exception as e:
        out.append(Violation("ownership.user_missing", f"query raised: {e!r}"))
    try:
        orphan_msgs = _query(
            db,
            "SELECT m.id, m.conversation_id FROM conversation_messages m "
            "LEFT JOIN conversations c ON c.id = m.conversation_id WHERE c.id IS NULL",
        )
        if orphan_msgs:
            out.append(Violation("referential.orphan_messages",
                                  f"{len(orphan_msgs)} message(s) without a conversation"))
    except Exception as e:
        out.append(Violation("referential.orphan_messages", f"query raised: {e!r}"))


# ── Sessions / state machine ────────────────────────────────────────────

def _check_sessions(runtime, db, out: list[Violation]) -> None:
    if runtime is None:
        return
    sessions = getattr(runtime, "sessions", {}) or {}

    active = getattr(runtime, "active_session_key", None)
    if active is not None and active not in sessions:
        out.append(Violation("runtime.active_session",
                              f"active_session_key {active!r} not in live sessions"))

    for key, session in sessions.items():
        cs = getattr(session, "cs", None)
        cid = getattr(session, "conversation_id", None)

        # Ownership: a live session must never be bound to a conversation its
        # effective user does not own. This is the kernel's core guarantee.
        if cid is not None and db is not None:
            try:
                row = db.get_conversation(cid)
            except Exception:
                row = None
            if row is None:
                out.append(Violation("session.dangling_conversation",
                                      f"session {key!r} bound to missing conversation {cid}"))
            else:
                owner = row["user_id"] if "user_id" in row.keys() else DEFAULT_USER_ID
                eff = _effective_user(runtime, key)
                if eff is not None and owner != eff:
                    out.append(Violation("session.ownership_crossed",
                                         f"session {key!r} (user {eff}) bound to "
                                         f"conversation {cid} owned by user {owner}"))

        if cs is None:
            continue

        # turn_priority must name a real participant.
        tp = getattr(cs, "turn_priority", None)
        if tp not in getattr(cs, "participants", {}):
            out.append(Violation("state.turn_priority",
                                 f"session {key!r} turn_priority {tp!r} is not a participant"))

        # Idle hygiene: when the turn is back with the user and nothing is
        # awaiting approval or form input, the phase-frame stack must be fully
        # unwound. A session mid-form (filling_tool_form / filling_command_form)
        # is legitimately suspended awaiting the user — forms persist across
        # restarts by design — so it is not idle.
        phases = (getattr(cs, "cache", {}) or {}).get("phases") or []
        awaiting_form = getattr(cs, "phase", None) in FORM_PHASES
        idle = tp == "user" and not awaiting_form and not _has_pending_approval(runtime, key)
        if idle and phases:
            out.append(Violation("state.leaked_phase_frame",
                                 f"session {key!r} idle but {len(phases)} phase frame(s) remain"))


def _effective_user(runtime, key: str):
    for attr in ("session_user_id",):
        fn = getattr(runtime, attr, None)
        if callable(fn):
            try:
                return fn(key)
            except Exception:
                return None
    session = (getattr(runtime, "sessions", {}) or {}).get(key)
    return getattr(session, "user_id", None)


def _has_pending_approval(runtime, key: str) -> bool:
    fn = getattr(runtime, "has_pending_approval", None)
    if callable(fn):
        try:
            return bool(fn(key))
        except Exception:
            return False
    return False


# ── Action ledger ─────────────────────────────────────────────────────

def _check_ledger(db, out: list[Violation]) -> None:
    """Recent ledger rows are well-formed: required columns present, origin in
    its enum, ok boolean, JSON columns valid JSON. Kept cheap (newest rows
    only) so the oracle stays fast inside the fuzzer loop."""
    if db is None or not hasattr(db, "get_ledger_rows"):
        return
    import json
    try:
        rows = db.get_ledger_rows(limit=50)
    except Exception as e:
        out.append(Violation("ledger.read", f"get_ledger_rows raised: {e!r}"))
        return
    for row in rows:
        rid = row.get("id")
        if not row.get("ts") or not row.get("action_type"):
            out.append(Violation("ledger.malformed", f"row {rid} missing ts/action_type"))
        if row.get("origin") not in {"user_enact", "agent_enact", "system"}:
            out.append(Violation("ledger.origin", f"row {rid} origin {row.get('origin')!r}"))
        if row.get("ok") not in (0, 1):
            out.append(Violation("ledger.ok", f"row {rid} ok={row.get('ok')!r}"))
        for col in ("args_json", "data_json"):
            value = row.get(col)
            if value is None:
                continue
            try:
                json.loads(value)
            except Exception:
                out.append(Violation("ledger.json", f"row {rid} {col} is not valid JSON"))


# ── Registry / threads ────────────────────────────────────────────────

def _check_registry(runtime, out: list[Violation]) -> None:
    registry = getattr(runtime, "tool_registry", None)
    if registry is None:
        return
    tools = set(getattr(registry, "tools", {}) or {})
    visible = getattr(registry, "visible_tool_names", None)
    if visible is not None:
        extra = set(visible) - tools
        if extra:
            out.append(Violation("registry.visible_not_registered",
                                 f"visible tools not in registry: {sorted(extra)}"))


def _check_threads(baseline: set[str] | None, out: list[Violation]) -> None:
    if baseline is None:
        return
    # Ephemeral worker threads the kernel spawns per submission; transient and
    # expected to drain, so don't flag them as leaks.
    transient_prefixes = ("repl-submit", "agent", "ThreadPoolExecutor", "asyncio")
    leaked = {
        name for name in (thread_names() - baseline)
        if not name.startswith(transient_prefixes)
    }
    if leaked:
        out.append(Violation("threads.leaked", f"new persistent threads: {sorted(leaked)}"))
