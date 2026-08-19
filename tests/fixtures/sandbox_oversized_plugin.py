"""Fixture: SDK code that asks a question whose honest answer will not fit.

Run in-process *and* in a subprocess by ``test_sandbox_oversized.py``, from
this one file, because the two boundaries used to disagree about what happens
next: in-process the answer became a failure, over a pipe ``protocol.encode``
raised out of ``runner_subprocess.send`` and took the whole box down.

``db.query`` is the route rather than anything contrived. It caps *rows* at
``DB_MAX_ROWS`` and says nothing about their size, which is the same mistake
``conv.read`` made — so twenty fat rows is a real, unpatched way to ask for
more than the wire carries.
"""

TABLE = "sandbox_oversized_probe"


def _fill(sdk, rows, width):
    """Twenty megabytes without twenty megabytes of Request traffic.

    Built inside sqlite with a recursive CTE, because sending the payload in
    as a parameter would put it through the very boundary under test on the
    way in.
    """
    sdk.db.define(f"CREATE TABLE IF NOT EXISTS {TABLE} ("
                  f"  id INTEGER PRIMARY KEY, blob TEXT)")
    sdk.db.write(f"DELETE FROM {TABLE}")
    sdk.db.write(
        f"INSERT INTO {TABLE} (blob) "
        f"SELECT hex(randomblob(?)) FROM ("
        f"  WITH RECURSIVE counter(x) AS ("
        f"    SELECT 1 UNION ALL SELECT x + 1 FROM counter WHERE x < ?)"
        f"  SELECT x FROM counter)",
        [width // 2, rows])


def ask_for_too_much(sdk, rows=20, width=1_000_000):
    """Ask, then report what came back — never raising past the runner."""
    _fill(sdk, rows, width)
    try:
        got = sdk.db.query(f"SELECT * FROM {TABLE}")
    except sdk.Failed as failed:
        return {"raised": True, "code": failed.result.code,
                "error": failed.error, "denied": isinstance(failed, sdk.Denied)}
    return {"raised": False, "rows": len(got)}


def still_usable_afterwards(sdk, rows=20, width=1_000_000):
    """The property the crash was really about: the box survives the refusal.

    A frontend is a resident box. When an oversized answer killed it, the poll
    loop had nothing to poll and the UI simply stopped — so what matters is not
    only that the failure is reported, but that the next Request still works.
    """
    _fill(sdk, rows, width)
    try:
        sdk.db.query(f"SELECT * FROM {TABLE}")
    except sdk.Failed:
        pass
    rows_back = sdk.db.query(f"SELECT COUNT(*) AS n FROM {TABLE}")
    return {"alive": True, "counted": rows_back[0]["n"]}
