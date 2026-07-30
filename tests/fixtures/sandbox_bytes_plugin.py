"""Fixture: SDK code that puts bytes through the database.

Loaded in-process *and* run in a subprocess by ``test_sandbox_bytes.py``, from
this one file, so any difference between the two runners is a difference in the
boundary rather than in the code.
"""

BLOB = b"\x00\x01\x02\xfe\xff"


def _ensure_table(sdk):
    """Create the scratch table this fixture owns."""
    sdk.db.define(
        "CREATE TABLE IF NOT EXISTS sandbox_bytes_probe ("
        "  key TEXT PRIMARY KEY,"
        "  payload BLOB"
        ")"
    )


def store_and_fetch(sdk, key="k"):
    """Write a BLOB and read it straight back."""
    _ensure_table(sdk)
    sdk.db.write(
        "INSERT OR REPLACE INTO sandbox_bytes_probe (key, payload) "
        "VALUES (?, ?)",
        [key, BLOB],
    )
    rows = sdk.db.query(
        "SELECT payload FROM sandbox_bytes_probe WHERE key = ?", [key])
    blob = rows[0]["payload"]
    return {"blob": blob, "length": len(blob)}


def store_and_report_type(sdk, key="typed"):
    """Ask sqlite what storage class the value actually landed in.

    A parameter that arrived as a tagged dict rather than as bytes would be
    bound as text, and ``typeof`` is the only thing that can tell the two
    apart once the value is back on this side.
    """
    _ensure_table(sdk)
    sdk.db.write(
        "INSERT OR REPLACE INTO sandbox_bytes_probe (key, payload) "
        "VALUES (?, ?)",
        [key, BLOB],
    )
    rows = sdk.db.query(
        "SELECT typeof(payload) AS kind FROM sandbox_bytes_probe "
        "WHERE key = ?", [key])
    return rows[0]["kind"]
