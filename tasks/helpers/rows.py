"""Reading a whole table's worth of rows through a capped Request.

``sdk.db.query`` answers at most ``DB_MAX_ROWS`` (500) rows and says so only by
handing back exactly that many. A long document produces more than 500 chunks,
so a task that reads them in one call gets a silently truncated answer — it
embeds the first 500 and reports success, and the tail of the document is
missing from the index with nothing anywhere to say so.

The cap is not a mistake to route around: it exists so a runaway query cannot
exhaust the parent's memory, and one page still crosses the wire at a time.
What this adds is the loop, in one place, so the two tasks that need every row
of a file's chunks do not each get it subtly wrong.

Stdlib-only and free of the SDK's shape beyond ``sdk.db.query``, so it costs
its importer nothing — a task declaring only this stays in-process.
"""

#: Matches the kernel's own ceiling. A smaller page would work and cost more
#: round trips; a larger one is silently clamped, which would make this loop
#: think it had reached the end when it had not.
PAGE = 500


def paged(sdk, sql: str, params=None, page: int = PAGE) -> list:
    """Every row ``sql`` matches, gathered a page at a time.

    ``sql`` must carry its own ``ORDER BY`` and must *not* carry ``LIMIT`` or
    ``OFFSET`` — those are appended here. A stable order is what makes paging
    meaningful: without one, two pages can overlap or skip rows entirely, and
    sqlite is under no obligation to be consistent between them.

    Stops on the first short page, which is the only honest end-of-data signal
    available: a full page might be the last one, so it costs one extra empty
    query to find out rather than guessing.
    """
    gathered = []
    offset = 0
    while True:
        rows = sdk.db.query(f"{sql} LIMIT {int(page)} OFFSET {int(offset)}",
                            list(params or []))
        gathered.extend(rows)
        if len(rows) < page:
            return gathered
        offset += page
