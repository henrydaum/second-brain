"""Database schema owned by public-web accounts and credits."""

import sqlite3


def setup(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_users (
            user_id TEXT PRIMARY KEY, session_id TEXT, ip_hash TEXT, created_at REAL, last_seen REAL,
            purchased_credits INTEGER DEFAULT 0, email TEXT, account_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_credit_ledger (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, cost INTEGER NOT NULL,
            free_amount INTEGER NOT NULL DEFAULT 0, paid_amount INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL, ts REAL NOT NULL, committed_at REAL, meta_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_credit_user_ts ON web_credit_ledger(user_id, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_credit_status_ts ON web_credit_ledger(status, ts)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_web_users_email ON web_users(email) WHERE email IS NOT NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_web_users_account ON web_users(account_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_web_users_ip ON web_users(ip_hash)")
    try:
        conn.execute("ALTER TABLE web_users ADD COLUMN account_config TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE TABLE IF NOT EXISTS web_auth_tokens (token TEXT PRIMARY KEY, email TEXT NOT NULL, created_at REAL NOT NULL, used_at REAL, anon_user_id TEXT)")
    try:
        conn.execute("ALTER TABLE web_auth_tokens ADD COLUMN anon_user_id TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_web_auth_tokens_email ON web_auth_tokens(email)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_promo_codes (
            code TEXT PRIMARY KEY, kind TEXT NOT NULL, credits INTEGER, max_uses INTEGER DEFAULT 1,
            uses INTEGER DEFAULT 0, created_at REAL NOT NULL, note TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_payments (
            id INTEGER PRIMARY KEY, stripe_event_id TEXT UNIQUE, email TEXT,
            amount_cents INTEGER, credits_granted INTEGER, ts REAL
        )
    """)
