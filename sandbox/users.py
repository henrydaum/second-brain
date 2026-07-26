"""Per-user scoping for ``db.query`` — whose rows, not which tables.

Ownership in Second Brain is *row*-scoped: ``conversations.user_id`` is the
source of truth, and only a couple of tables carry the column at all. So the
security axis is not "which tables may this read" but "whose rows".

Reads stay broad on purpose — a plugin that reads everything still cannot send
anything anywhere, because egress is gated. What is narrowed is identity.

**Why not real SQL views.** A view is the obvious answer and SQLite cannot
give it: views take no parameters, so ``my_conversations`` would have to be
created per user, per connection, and connections are shared. Rewriting
arbitrary SQL is the other obvious answer and it is worse — a fragile parser
standing where a security boundary should be.

So the honest version: sandboxed code writes a **virtual table name** the
kernel owns. ``my_conversations`` expands to a filtered subquery with the
session's user id inlined (an integer the kernel controls, never caller
input). Reading the base table directly is refused, with a message naming the
virtual one — the same teaching-error approach the validator takes.
"""

from __future__ import annotations

import re

# Tables carrying an owner, and the column that carries it.
USER_SCOPED = {
    "conversations": "user_id",
    "action_ledger": "user_id",
}

# Never readable through db.query, whatever the query looks like.
FORBIDDEN_COLUMNS = ("password_hash",)

VIRTUAL_PREFIX = "my_"


class ScopeError(Exception):
    """A query reached for rows it may not have."""


def _mentions(sql: str, word: str) -> bool:
    """Whether a bare identifier appears in the SQL."""
    return re.search(rf"\b{re.escape(word)}\b", sql, re.IGNORECASE) is not None


def virtual_names() -> dict:
    """The virtual table name for each user-scoped table."""
    return {f"{VIRTUAL_PREFIX}{table}": table for table in USER_SCOPED}


def scope_sql(sql: str, params, user_id):
    """Expand virtual table names and refuse cross-user reads.

    Returns ``(sql, params)`` ready to execute, or raises :class:`ScopeError`
    with a message the author can act on.
    """
    for column in FORBIDDEN_COLUMNS:
        if _mentions(sql, column):
            raise ScopeError(
                f"{column} is never readable through a Request")

    scoped = sql
    for virtual, table in virtual_names().items():
        if not _mentions(scoped, virtual):
            continue
        if user_id is None:
            raise ScopeError(
                f"{virtual} needs a user, and this execution has none")
        owner = USER_SCOPED[table]
        # The id is an integer the kernel supplies, never caller input, so
        # inlining it cannot shift the caller's parameter positions.
        subquery = (f"(SELECT * FROM {table} "
                    f"WHERE {owner} = {int(user_id)})")
        scoped = re.sub(rf"\b{re.escape(virtual)}\b", subquery, scoped,
                        flags=re.IGNORECASE)

    for table in USER_SCOPED:
        if _mentions(scoped, table) and not _mentions(sql, f"{VIRTUAL_PREFIX}{table}"):
            raise ScopeError(
                f"{table} holds other people's rows; "
                f"read {VIRTUAL_PREFIX}{table} instead")

    return scoped, params
