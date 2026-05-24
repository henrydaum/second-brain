"""Simple integer credits for public web prompts and completed renders."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid

from config.config_data import DEFAULT_WEB_CREDITS
from events.event_bus import bus
from events.event_channels import CREDITS_CHANGED, CREDIT_ACTION_DENIED
from plugins.BaseService import BaseService


class CreditDenied(RuntimeError):
    """A web compute action could not be funded."""

    def __init__(self, payload: dict):
        self.payload = payload
        super().__init__(str(payload.get("reason") or "insufficient_credits"))


class CreditsService(BaseService):
    """Single authority for public web balances and compute reservations."""

    model_name = "credits"
    shared = True

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._sessions: dict[str, str] = {}
        self._lock = threading.RLock()

    def _load(self) -> bool:
        self.loaded = True
        return True

    def unload(self):
        self._sessions.clear()
        self.loaded = False

    def policy(self) -> dict:
        raw = self.config.get("web_credits") if isinstance(self.config, dict) else {}
        return {key: {**defaults, **((raw or {}).get(key) or {})} for key, defaults in DEFAULT_WEB_CREDITS.items()}

    def pack(self) -> dict:
        return self.policy()["pack"]

    def ai_cost(self) -> int:
        return max(0, int(self.policy()["costs"]["ai_prompt"]))

    def render_cost(self) -> int:
        return max(0, int(self.policy()["costs"]["uncached_render"]))

    @staticmethod
    def _anon_id(session_id: str, ip: str) -> str:
        return hashlib.sha256(f"{ip}|{session_id}".encode()).hexdigest()[:24]

    @staticmethod
    def _ip_hash(ip: str) -> str:
        return hashlib.sha256(ip.encode()).hexdigest()[:16] if ip else ""

    def bind_web_session(self, db, session_key: str, session_id: str, ip: str = "", account_id: str = "") -> str:
        now = time.time()
        with self._lock, db.lock:
            row = db.conn.execute("SELECT user_id FROM web_users WHERE account_id = ?", (account_id,)).fetchone() if account_id else None
            uid = row["user_id"] if row else self._anon_id(session_id, ip)
            db.conn.execute(
                "INSERT INTO web_users (user_id, session_id, ip_hash, created_at, last_seen, purchased_credits) VALUES (?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(user_id) DO UPDATE SET session_id=excluded.session_id, ip_hash=excluded.ip_hash, last_seen=excluded.last_seen",
                (uid, session_id, self._ip_hash(ip), now, now),
            )
            db.conn.commit()
            self._sessions[session_key] = uid
        return uid

    def _availability(self, db, uid: str, now: float | None = None) -> tuple[int, int]:
        now = now or time.time()
        p = self.policy()
        sums = []
        for since in (now - 5 * 3600, now - 7 * 24 * 3600):
            sums.append(db.conn.execute(
                "SELECT COALESCE(SUM(free_amount),0) n FROM web_credit_ledger WHERE user_id=? AND ts>=? AND status IN ('reserved','committed')",
                (uid, since),
            ).fetchone()["n"])
        free = min(max(0, int(p["free"]["five_hours"]) - int(sums[0])), max(0, int(p["free"]["week"]) - int(sums[1])))
        row = db.conn.execute("SELECT COALESCE(purchased_credits,0) n FROM web_users WHERE user_id=?", (uid,)).fetchone()
        paid = int(row["n"]) if row else 0
        reserved = db.conn.execute(
            "SELECT COALESCE(SUM(paid_amount),0) n FROM web_credit_ledger WHERE user_id=? AND status='reserved'",
            (uid,),
        ).fetchone()["n"]
        return free, max(0, paid - int(reserved))

    def _next_refill_seconds(self, db, uid: str, now: float | None = None) -> int | None:
        now = now or time.time()
        waits = []
        limits = self.policy()["free"]
        for span, cap in ((5 * 3600, limits["five_hours"]), (7 * 24 * 3600, limits["week"])):
            row = db.conn.execute(
                "SELECT COALESCE(SUM(free_amount),0) n, MIN(ts) oldest FROM web_credit_ledger "
                "WHERE user_id=? AND ts>=? AND status IN ('reserved','committed') AND free_amount>0",
                (uid, now - span),
            ).fetchone()
            if int(row["n"]) >= int(cap) and row["oldest"] is not None:
                waits.append(max(0, int(float(row["oldest"]) + span - now)))
        return max(waits) if waits else None

    def _snapshot_locked(self, db, session_key: str, uid: str) -> dict:
        free, paid = self._availability(db, uid)
        row = db.conn.execute("SELECT email, account_id FROM web_users WHERE user_id=?", (uid,)).fetchone()
        pack = self.pack()
        return {
            "session_key": session_key, "signed_in": bool(row and row["account_id"]), "email": row["email"] if row else None,
            "free_remaining": free, "purchased_remaining": paid, "total_available": free + paid,
            "next_refill_seconds": self._next_refill_seconds(db, uid) if not free else None,
            "pack": {"credits": int(pack["credits"]), "price_cents": int(pack["price_cents"])},
        }

    def snapshot(self, db, session_key: str) -> dict:
        with self._lock, db.lock:
            uid = self._sessions.get(session_key)
            return self._snapshot_locked(db, session_key, uid) if uid else {}

    def reserve(self, db, session_key: str, kind: str, cost: int, metadata: dict | None = None) -> str | None:
        cost = max(0, int(cost))
        if not cost:
            return None
        payload = None
        with self._lock, db.lock:
            uid = self._sessions.get(session_key)
            if not uid:
                return None
            now = time.time()
            free, paid = self._availability(db, uid, now)
            if free + paid < cost:
                payload = {**self._snapshot_locked(db, session_key, uid), "required": cost, "action": kind, "reason": "insufficient_credits"}
            if payload is None:
                free = min(cost, free)
                paid = cost - free
                rid = uuid.uuid4().hex
                db.conn.execute(
                    "INSERT INTO web_credit_ledger (id,user_id,kind,cost,free_amount,paid_amount,status,ts,meta_json) VALUES (?,?,?,?,?,?, 'reserved', ?, ?)",
                    (rid, uid, kind, cost, free, paid, now, json.dumps(metadata or {}, separators=(",", ":"))),
                )
                db.conn.commit()
                return rid
        bus.emit(CREDIT_ACTION_DENIED, payload)
        raise CreditDenied(payload)

    def commit(self, db, reservation_id: str | None, actual_cost: int | None = None, session_key: str = "") -> None:
        if not reservation_id:
            return
        visible = None
        with self._lock, db.lock:
            row = db.conn.execute("SELECT * FROM web_credit_ledger WHERE id=? AND status='reserved'", (reservation_id,)).fetchone()
            if not row:
                return
            cost = int(row["cost"]) if actual_cost is None else max(0, min(int(row["cost"]), int(actual_cost)))
            free = min(int(row["free_amount"]), cost)
            paid = cost - free
            db.conn.execute("UPDATE web_credit_ledger SET cost=?, free_amount=?, paid_amount=?, status='committed', committed_at=? WHERE id=?", (cost, free, paid, time.time(), reservation_id))
            if paid:
                db.conn.execute("UPDATE web_users SET purchased_credits=MAX(0,purchased_credits-?) WHERE user_id=?", (paid, row["user_id"]))
            db.conn.commit()
            if session_key:
                visible = self._snapshot_locked(db, session_key, row["user_id"])
        if visible:
            bus.emit(CREDITS_CHANGED, visible)

    def release(self, db, reservation_id: str | None, session_key: str = "") -> None:
        if not reservation_id:
            return
        visible = None
        with self._lock, db.lock:
            row = db.conn.execute("SELECT user_id FROM web_credit_ledger WHERE id=? AND status='reserved'", (reservation_id,)).fetchone()
            if not row:
                return
            db.conn.execute("UPDATE web_credit_ledger SET status='released', committed_at=? WHERE id=?", (time.time(), reservation_id))
            db.conn.commit()
            if session_key:
                visible = self._snapshot_locked(db, session_key, row["user_id"])
        if visible:
            bus.emit(CREDITS_CHANGED, visible)

    def grant(self, db, user_id: str, amount: int, kind: str, session_key: str = "") -> None:
        amount = max(0, int(amount))
        if not amount:
            return
        with self._lock, db.lock:
            db.conn.execute("UPDATE web_users SET purchased_credits=COALESCE(purchased_credits,0)+? WHERE user_id=?", (amount, user_id))
            db.conn.execute(
                "INSERT INTO web_credit_ledger (id,user_id,kind,cost,free_amount,paid_amount,status,ts,committed_at,meta_json) VALUES (?,?,?,-?,0,-?,'committed',?,?, '{}')",
                (uuid.uuid4().hex, user_id, kind, amount, amount, time.time(), time.time()),
            )
            db.conn.commit()
            visible = self._snapshot_locked(db, session_key, user_id) if session_key else None
        if visible:
            bus.emit(CREDITS_CHANGED, visible)

    def render_authorizer(self, db, session_key: str, allow_prompt_overrun: bool = False):
        """Reserve a flat cache-miss fee; accepted renders always finish."""
        def authorize():
            if allow_prompt_overrun:
                with self._lock, db.lock:
                    uid = self._sessions.get(session_key)
                    active = uid and db.conn.execute(
                        "SELECT 1 FROM web_credit_ledger WHERE user_id=? AND kind='ai_prompt' AND status='reserved' LIMIT 1",
                        (uid,),
                    ).fetchone()
                    if active and sum(self._availability(db, uid)) < self.render_cost():
                        return None
            rid = self.reserve(db, session_key, "render", self.render_cost())
            return lambda success: (self.commit if success else self.release)(db, rid, session_key=session_key)
        return authorize


def build_services(config: dict) -> dict:
    return {"credits": CreditsService(config or {})}
