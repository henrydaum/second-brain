"""
Helper script to remove the 'source' column from the 'conversations' table.

SQLite doesn't support ALTER TABLE ... DROP COLUMN (before 3.35.0), so this
script recreates the table without the unwanted column and copies data over.

Usage:
    python drop_column.py [--db-path PATH]

If no --db-path is given, it uses the default location from config.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent))


def get_default_db_path() -> Path:
    from paths import DATA_DIR
    return DATA_DIR / "database.db"


def drop_column(db_path: Path):
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    print(f"Database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Check that the column actually exists
    cur = conn.execute("PRAGMA table_info(conversations)")
    columns = {row["name"] for row in cur.fetchall()}

    if "source" not in columns:
        print("Column 'source' does not exist in 'conversations' — nothing to do.")
        conn.close()
        return

    print(f"Current columns: {sorted(columns)}")
    print("Removing 'source' column...")

    # SQLite 3.35.0+ supports ALTER TABLE DROP COLUMN natively.
    # Use it if available, otherwise fall back to the table-rebuild approach.
    sqlite_version = tuple(int(x) for x in sqlite3.sqlite_version.split("."))

    if sqlite_version >= (3, 35, 0):
        conn.execute("ALTER TABLE conversations DROP COLUMN source")
    else:
        # Classic rebuild: create temp table → copy data → drop old → rename
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN TRANSACTION")
        conn.execute("""
            CREATE TABLE conversations_backup (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT,
                created_at  REAL,
                updated_at  REAL
            )
        """)
        conn.execute("""
            INSERT INTO conversations_backup (id, title, created_at, updated_at)
            SELECT id, title, created_at, updated_at FROM conversations
        """)
        conn.execute("DROP TABLE conversations")
        conn.execute("ALTER TABLE conversations_backup RENAME TO conversations")
        conn.execute("COMMIT")
        conn.execute("PRAGMA foreign_keys = ON")

    conn.commit()

    # Verify
    cur = conn.execute("PRAGMA table_info(conversations)")
    new_columns = {row["name"] for row in cur.fetchall()}
    print(f"New columns:     {sorted(new_columns)}")

    if "source" not in new_columns:
        print("Done — 'source' column removed successfully.")
    else:
        print("WARNING: Column 'source' still present. Something went wrong.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove the 'source' column from the conversations table."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to the SQLite database file. Defaults to the app's standard location.",
    )
    args = parser.parse_args()

    path = args.db_path or get_default_db_path()

    print("=" * 50)
    print("Drop 'source' column from conversations")
    print("=" * 50)

    response = input(f"This will modify: {path}\nContinue? [y/N] ").strip().lower()
    if response != "y":
        print("Aborted.")
        sys.exit(0)

    drop_column(path)
