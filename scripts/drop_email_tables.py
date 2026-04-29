"""
One-off cleanup: drop the unused emails / email_threads tables left over
from the old task_check_inbox SQL mirror. Safe to re-run.

Usage:
    python scripts/drop_email_tables.py
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_DIR

DB_PATH = DATA_DIR / "database.db"
TABLES = ["emails", "email_threads"]


def main() -> int:
    if not DB_PATH.exists():
        print(f"No database at {DB_PATH} — nothing to do.")
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        for tbl in TABLES:
            existed = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (tbl,),
            ).fetchone()
            cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            print(f"{'dropped' if existed else 'absent '} {tbl}")
        conn.commit()
        cur.execute("VACUUM")
    finally:
        conn.close()
    print(f"Done. ({DB_PATH})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
