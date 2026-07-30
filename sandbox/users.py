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


# Tables the kernel owns, and the Request that maintains each properly.
#
# Kernel table *rows* are writable through ``db.write``; kernel table *schemas*
# are not. That is the line, and it is a different line than the one this
# module first drew.
#
# **Why rows are fine.** Changing data cannot change how the kernel works —
# only changing structure can. Every query the kernel issues keeps working
# against edited rows and stops working against a dropped column, so structure
# is where the breakage lives. The two objections to opening rows both turned
# out to be wrong: ``conversation_messages`` declares ``ON DELETE CASCADE`` and
# the connection runs with ``PRAGMA foreign_keys = ON``, so a raw
# ``DELETE FROM conversations`` cascades exactly as ``conv.delete`` does — which
# is itself a bare DELETE — and nothing is orphaned. Meanwhile the wall's real
# cost was ordinary bookkeeping: not every kernel *column* has a Request behind
# it (``conversations.last_title_check_message_count`` is a high-water mark with
# no ``conv.*`` verb), so a task keeping one had to invent a shadow table
# mirroring rows it could already read.
#
# The named Request is still the better route where one exists: it carries the
# owner check, it emits the bus event frontends redraw from, and it is what the
# ledger records meaningfully. The mapping below is what an author gets pointed
# at, not a wall.
#
# **Why the keyword check is sound here and was not for the shell.**
# ``Database.execute_write`` uses ``conn.execute``, which runs exactly one
# statement — a ``;``-chained script raises. So reading the leading keyword is
# not racing against chaining, quoting or substitution the way
# ``tool_run_command``'s whitelist was. One statement, first token, decidable.
#
# **Three things stay refused**, and none of them is "a kernel table":
#
#   - ``password_hash`` — unwritable for the same reason it is unreadable.
#     There is no bookkeeping reason to touch a credential column, so the
#     carve-out costs nothing and the accident it prevents is unrecoverable.
#   - ``sqlite_master`` / ``sqlite_schema`` and ``PRAGMA`` — because DDL is not
#     only spelled CREATE. ``PRAGMA writable_schema=ON`` followed by an
#     ``UPDATE sqlite_master SET sql=…`` is schema surgery wearing DML's
#     clothes, and without this the rule would ship with a documented bypass.
#   - DDL naming a kernel table — the actual point.
#
# **Known gap, deliberately left.** A row write has no owner check. Reads solve
# that with the ``my_`` virtual name, and writes cannot: SQLite will not UPDATE
# a subquery, so there is nothing for the trick to expand into. Today every
# frontend is single-user and it costs nothing; it becomes real the day a
# ``per_user`` frontend is, and closing it then means inspecting WHERE clauses —
# the fragile-parser shape this module otherwise refuses to build.
KERNEL_TABLES = {
    "users": "sdk.user.write",
    "conversations": "sdk.conv.* (create, set_title, delete, ...)",
    "conversation_messages": "sdk.conv.append",
    "action_ledger": "sdk.ledger.record",
    "files": "sdk.file.register",
    "registered_tasks": "sdk.task.* (enqueue, pause, trigger, ...)",
    "task_queue": "sdk.task.enqueue",
    "task_runs": "sdk.task.* (status, reset, ...)",
}


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


# Statements that change one table's structure. Refused when they name a kernel
# table, allowed otherwise — ``db.define`` exists to create things, and a
# plugin's own table is nobody else's business.
DDL_KEYWORDS = frozenset({
    "CREATE", "DROP", "ALTER", "RENAME", "REINDEX",
})

# Statements that act on the database *file* rather than on any table in it, and
# are therefore refused whether or not a kernel table is named. Gating these on
# a table name is a mistake worth spelling out: ``ATTACH DATABASE '/etc/x.db'``
# mentions no table at all, so a per-table check waves it straight through —
# and what it actually does is open an arbitrary file and expose it to SQL,
# which is filesystem access spelled in a language the filesystem rules do not
# read. ``VACUUM`` rewrites the whole database, which is not a plugin's call to
# make on a store everything else is using.
FILE_LEVEL_KEYWORDS = frozenset({"ATTACH", "DETACH", "VACUUM"})

# The schema table under both of its names, plus the pragma that makes it
# writable. Refused in any statement, kernel table named or not: there is no
# legitimate plugin reason to touch either, and together they are the way DML
# becomes DDL.
_SCHEMA_NAMES = ("sqlite_master", "sqlite_schema")

_LEADING = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/|\s)*([A-Za-z_]+)", re.DOTALL)


def leading_keyword(sql: str) -> str:
    """The statement's first bare word, uppercased, or "".

    Comments before it are skipped, since a statement may legitimately open
    with one. Only one statement can arrive here — ``conn.execute`` refuses a
    chained script — so this word describes the whole of what will run.
    """
    match = _LEADING.match(sql or "")
    return match.group(1).upper() if match else ""


def touches_kernel_table(sql: str) -> str:
    """The first kernel table a statement mentions, or ""."""
    for table in KERNEL_TABLES:
        if _mentions(sql, table):
            return table
    return ""


def is_kernel_delete(sql: str) -> str:
    """The kernel table a DELETE would empty rows from, or "".

    Separated out because the policy function asks a different question than
    this module does: not "may this be asked at all" but "should a person be
    asked first". A delete is the one row write that cannot be undone by
    writing again.
    """
    if leading_keyword(sql) != "DELETE":
        return ""
    return touches_kernel_table(sql)


def scope_write(sql: str):
    """Check a statement on its way to ``db.write`` or ``db.define``.

    Returns ``sql`` unchanged, or raises :class:`ScopeError`. Kernel table rows
    are writable and kernel table schemas are not — see :data:`KERNEL_TABLES`
    for the argument, and prefer the named Request where one exists, because
    that is the route carrying the owner check and the bus event.

    Deliberately an *identifier and first-token* check rather than a statement
    parser. Which names a statement mentions and which word it starts with are
    both answerable by looking; what an arbitrary statement does is not, and a
    fragile SQL parser standing where a security boundary should be is the
    thing this module refuses to build.
    """
    for column in FORBIDDEN_COLUMNS:
        if _mentions(sql, column):
            raise ScopeError(
                f"{column} is never writable through a Request")

    keyword = leading_keyword(sql)
    if keyword == "PRAGMA":
        raise ScopeError(
            "PRAGMA is not available through a Request: it configures the "
            "database itself, and one setting (writable_schema) turns an "
            "ordinary UPDATE into a schema change")
    if keyword in FILE_LEVEL_KEYWORDS:
        raise ScopeError(
            f"{keyword} acts on the database file rather than on its rows and "
            f"is not available through a Request; reach files with sdk.fs")
    for name in _SCHEMA_NAMES:
        if _mentions(sql, name):
            raise ScopeError(
                f"{name} is the schema itself and is never writable through a "
                f"Request; use sdk.db.define to create your own tables")

    if keyword in DDL_KEYWORDS and (table := touches_kernel_table(sql)):
        raise ScopeError(
            f"{keyword} would change the structure of {table}, a kernel "
            f"table — its rows are writable but its schema is not. Use "
            f"{KERNEL_TABLES[table]} to change data, or sdk.db.define to "
            f"create a table of your own")
    return sql
