from __future__ import annotations

import time
from pathlib import Path

import pytest

from config.config_data import DEFAULT_WEB_CREDITS
from config import config_manager
from pipeline.database import Database
from plugins.services.service_credits import CreditDenied, CreditsService


@pytest.fixture
def db(request):
    path = Path(f".credits_test_{request.node.name}.db")
    for p in (path, Path(str(path) + "-shm"), Path(str(path) + "-wal")):
        p.unlink(missing_ok=True)
    value = Database(str(path))
    yield value
    value.conn.close()
    for p in (path, Path(str(path) + "-shm"), Path(str(path) + "-wal")):
        p.unlink(missing_ok=True)


def _bound(db, config=None):
    svc = CreditsService(config or {})
    key = "web:test"
    uid = svc.bind_web_session(db, key, "test", "127.0.0.1")
    return svc, key, uid


def test_default_policy_is_flat_render_and_ten_credit_prompt():
    assert DEFAULT_WEB_CREDITS["costs"] == {"ai_prompt": 10, "uncached_render": 1}
    assert DEFAULT_WEB_CREDITS["free"] == {"five_hours": 60, "week": 600}
    assert DEFAULT_WEB_CREDITS["pack"]["credits"] == 1000
    assert DEFAULT_WEB_CREDITS["pack"]["price_cents"] == 299


def test_web_credits_are_saved_in_core_config(tmp_path):
    path = tmp_path / "config.json"
    config = config_manager.load(str(path))
    assert config["web_credits"] == DEFAULT_WEB_CREDITS
    assert "web_credits" in path.read_text()


def test_free_is_spent_before_purchased_and_action_can_split(db):
    svc, key, uid = _bound(db, {"web_credits": {"free": {"five_hours": 3, "week": 4}}})
    svc.grant(db, uid, 5, "purchase")
    rid = svc.reserve(db, key, "ai_prompt", 4)
    svc.commit(db, rid, session_key=key)
    row = db.conn.execute("SELECT free_amount, paid_amount FROM web_credit_ledger WHERE id=?", (rid,)).fetchone()
    assert (row["free_amount"], row["paid_amount"]) == (3, 1)
    assert svc.snapshot(db, key)["purchased_remaining"] == 4


def test_five_hour_free_window_rolls_off_while_weekly_use_remains(db):
    svc, key, uid = _bound(db)
    with db.lock:
        db.conn.execute(
            "INSERT INTO web_credit_ledger (id,user_id,kind,cost,free_amount,paid_amount,status,ts,committed_at,meta_json) VALUES ('old',?,'render',60,60,0,'committed',?,?, '{}')",
            (uid, time.time() - 6 * 3600, time.time() - 6 * 3600),
        )
        db.conn.commit()
    assert svc.snapshot(db, key)["free_remaining"] == 60


def test_snapshot_reports_five_hour_refill_when_free_is_empty(db):
    svc, key, uid = _bound(db, {"web_credits": {"free": {"five_hours": 2, "week": 20}}})
    now = time.time()
    with db.lock:
        db.conn.execute(
            "INSERT INTO web_credit_ledger (id,user_id,kind,cost,free_amount,paid_amount,status,ts,committed_at,meta_json) VALUES ('spent',?,'render',2,2,0,'committed',?,?, '{}')",
            (uid, now - 60, now - 60),
        )
        db.conn.commit()
    remaining = svc.snapshot(db, key)["next_refill_seconds"]
    assert 5 * 3600 - 65 <= remaining <= 5 * 3600 - 55


def test_week_limit_controls_refill_even_after_five_hour_window(db):
    svc, key, uid = _bound(db, {"web_credits": {"free": {"five_hours": 2, "week": 2}}})
    now = time.time()
    with db.lock:
        db.conn.execute(
            "INSERT INTO web_credit_ledger (id,user_id,kind,cost,free_amount,paid_amount,status,ts,committed_at,meta_json) VALUES ('weekly',?,'render',2,2,0,'committed',?,?, '{}')",
            (uid, now - 6 * 3600, now - 6 * 3600),
        )
        db.conn.commit()
    snap = svc.snapshot(db, key)
    assert snap["free_remaining"] == 0
    assert 6 * 24 * 3600 + 17 * 3600 <= snap["next_refill_seconds"] <= 6 * 24 * 3600 + 19 * 3600


def test_active_prompt_reservation_prevents_concurrent_overdraw(db):
    svc, key, _ = _bound(db, {"web_credits": {"free": {"five_hours": 10, "week": 10}}})
    rid = svc.reserve(db, key, "ai_prompt", 10)
    with pytest.raises(CreditDenied):
        svc.reserve(db, key, "ai_prompt", 10)
    svc.release(db, rid, session_key=key)
    assert svc.snapshot(db, key)["free_remaining"] == 10


def test_uncached_render_is_one_credit_and_cannot_begin_at_zero(db):
    svc, key, _ = _bound(db, {"web_credits": {"free": {"five_hours": 1, "week": 1}}})
    finish = svc.render_authorizer(db, key)()
    finish(True)
    assert svc.snapshot(db, key)["total_available"] == 0
    with pytest.raises(CreditDenied):
        svc.render_authorizer(db, key)()


def test_agent_render_completes_when_prompt_uses_last_credits(db):
    svc, key, _ = _bound(db, {"web_credits": {"free": {"five_hours": 10, "week": 10}}})
    prompt = svc.reserve(db, key, "ai_prompt", 10)
    assert svc.render_authorizer(db, key, allow_prompt_overrun=True)() is None
    svc.commit(db, prompt, session_key=key)
    assert svc.snapshot(db, key)["total_available"] == 0
