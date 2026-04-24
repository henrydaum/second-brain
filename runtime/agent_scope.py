"""
Agent scope — per-profile lens over the shared database and tool registry.

An agent profile (stored in ``llm_profiles`` config) can declare optional
restrictions on which tables it can read and which tools it can call.
This module turns those declarations into:

    - a ``ScopedDatabase`` that runs queries against an in-memory main
      schema with TEMP VIEWs redirecting to the real DB (attached read-only
      as ``source``), plus a SQLite authorizer that denies top-level reads
      of the ``source`` schema so scoped queries can only reach data
      through the views we create;
    - a ``ToolRegistry`` view that only exposes the allowed tools.

Omitting both ``_allow`` and ``_deny`` on a profile means no restriction —
that profile is a fully-capable agent, identical to today's behavior.
"""

import inspect
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass

from agent.tool_registry import ToolRegistry

logger = logging.getLogger("AgentScope")
_CALL_TOOL_RE = re.compile(r'context\.call_tool\(\s*["\']([^"\']+)["\']')

# SQLite authorizer action code (stable public SQLite C API value).
_SQLITE_READ = 20
_SOURCE_SCHEMA = "source"


@dataclass
class AgentScope:
    profile_name: str
    prompt_suffix: str = ""
    tools_allow: set[str] | None = None
    tools_deny: set[str] | None = None
    tables_allow: list | None = None  # list[str | dict{name, sql}]
    tables_deny: set[str] | None = None

    @property
    def has_tool_filter(self) -> bool:
        return self.tools_allow is not None or self.tools_deny is not None

    @property
    def has_table_filter(self) -> bool:
        return self.tables_allow is not None or self.tables_deny is not None


def load_scope(profile_name: str, config: dict) -> AgentScope:
    """Parse a profile's restriction fields into an ``AgentScope``.

    Raises ``ValueError`` if the profile sets both _allow and _deny for
    the same resource kind (tools or tables).
    """
    profile = config.get("llm_profiles", {}).get(profile_name, {}) or {}

    tools_allow = profile.get("tools_allow")
    tools_deny = profile.get("tools_deny")
    if tools_allow is not None and tools_deny is not None:
        raise ValueError(
            f"Profile '{profile_name}' sets both tools_allow and tools_deny — pick one."
        )

    tables_allow = profile.get("tables_allow")
    tables_deny = profile.get("tables_deny")
    if tables_allow is not None and tables_deny is not None:
        raise ValueError(
            f"Profile '{profile_name}' sets both tables_allow and tables_deny — pick one."
        )

    return AgentScope(
        profile_name=profile_name,
        prompt_suffix=str(profile.get("prompt_suffix") or ""),
        tools_allow=set(tools_allow) if tools_allow is not None else None,
        tools_deny=set(tools_deny) if tools_deny is not None else None,
        tables_allow=list(tables_allow) if tables_allow is not None else None,
        tables_deny=set(tables_deny) if tables_deny is not None else None,
    )


# ── Scoped database ──────────────────────────────────────────────────

class ScopedDatabase:
    """Read-only lens over a ``Database``.

    The connection has an empty in-memory main schema with only the views
    the profile is allowed to see; the real DB is attached read-only as
    ``source``. Named-view SQL in ``tables_allow`` must reference base
    tables via ``source.<name>`` — explicit qualification avoids circular
    references when a view shadows a same-named base table.

    A SQLite authorizer blocks top-level reads of the ``source`` schema,
    so direct ``SELECT * FROM source.foo`` is denied; reads inside a view
    body carry the view name in the 5th authorizer arg and are allowed.

    Every non-query attribute falls through to the base database so
    writes, conversation persistence, and internal helpers keep working
    unchanged.
    """

    def __init__(self, base_db, scope: AgentScope):
        if not scope.has_table_filter:
            raise ValueError("ScopedDatabase should only be built when tables are restricted.")
        self._base = base_db
        self._scope = scope
        self.lock = threading.Lock()  # matches Database.lock for tools that reach in
        # uri=True enables URI filename handling for ATTACH so we can open
        # the real DB read-only. Normalize backslashes for Windows-safe URIs.
        self._conn = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        normalized_path = str(base_db.db_path).replace("\\", "/")
        self._conn.execute(
            f"ATTACH DATABASE 'file:{normalized_path}?mode=ro' AS {_SOURCE_SCHEMA}"
        )
        self._visible_names: set[str] = set()
        self._build_views()
        self._install_authorizer()

    # Fall-through for everything else (writes, helpers, etc.)
    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def conn(self):
        # Some tool code reaches in for .conn directly — return the scoped one.
        return self._conn

    @property
    def db_path(self):
        return self._base.db_path

    def _build_views(self):
        """Create the views that form the scoped main schema."""
        cur = self._conn.execute(
            f"SELECT name FROM {_SOURCE_SCHEMA}.sqlite_master WHERE type IN ('table','view') ORDER BY name"
        )
        all_names = [r[0] for r in cur.fetchall()]

        if self._scope.tables_allow is not None:
            for entry in self._scope.tables_allow:
                if isinstance(entry, str):
                    if entry not in all_names:
                        logger.warning(f"tables_allow references missing object: {entry!r}")
                        continue
                    if not _is_safe_identifier(entry):
                        logger.warning(f"tables_allow bare name is not a safe identifier: {entry!r}")
                        continue
                    # Passthrough view in main -> source.<name>
                    self._create_view(entry, f"SELECT * FROM {_SOURCE_SCHEMA}.{entry}")
                elif isinstance(entry, dict):
                    name = entry.get("name")
                    sql = entry.get("sql")
                    if not name or not sql:
                        logger.warning(f"tables_allow entry missing name/sql: {entry!r}")
                        continue
                    if not _is_safe_identifier(name):
                        logger.warning(f"tables_allow entry has invalid name: {name!r}")
                        continue
                    self._create_view(name, sql)
                else:
                    logger.warning(f"Unrecognized tables_allow entry: {entry!r}")
        else:
            # deny list: create passthrough views for every source table except the denied ones.
            denied = self._scope.tables_deny or set()
            for name in all_names:
                if name in denied:
                    continue
                if not _is_safe_identifier(name):
                    continue
                # Skip SQLite-internal tables (e.g. sqlite_sequence) — can't
                # create views with reserved names.
                if name.startswith("sqlite_"):
                    continue
                self._create_view(name, f"SELECT * FROM {_SOURCE_SCHEMA}.{name}")

    def _create_view(self, name: str, sql: str):
        # TEMP views can reference objects in attached databases; regular
        # views cannot. Both are per-connection, so no schema pollution.
        try:
            self._conn.execute(f"CREATE TEMP VIEW {name} AS {sql}")
            self._visible_names.add(name)
        except sqlite3.Error as e:
            logger.error(f"Could not create scoped view {name!r}: {e}")

    def _install_authorizer(self):
        def authorizer(action, arg1, arg2, db_name, trigger_or_view):
            if action != _SQLITE_READ:
                return sqlite3.SQLITE_OK
            # Reads inside a view's body carry the view name in the 5th
            # arg and must be allowed — the view is the scope filter.
            if trigger_or_view is not None:
                return sqlite3.SQLITE_OK
            # Top-level reads can only hit our in-memory main schema and
            # sqlite_master. Reads on the attached source DB are denied.
            if db_name == _SOURCE_SCHEMA:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self._conn.set_authorizer(authorizer)

    def query(self, sql: str, max_rows: int = 25) -> dict:
        """Scoped read-only query. Same contract as ``Database.query``."""
        normalized = " ".join(sql.strip().split()).lower()
        if not (normalized.startswith("select") or normalized.startswith("pragma")):
            raise ValueError("Only SELECT and PRAGMA statements are allowed.")

        with self.lock:
            try:
                cur = self._conn.execute(sql)
            except sqlite3.DatabaseError as e:
                msg = str(e)
                if "not authorized" in msg.lower() or "is prohibited" in msg.lower():
                    # Reshape authorizer denials into OperationalError with a
                    # clearer message so sql_query's error path surfaces a
                    # useful hint about the scope filter.
                    raise sqlite3.OperationalError(
                        f"Access denied by agent scope: {msg}"
                    ) from e
                raise
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows + 1)

            truncated = len(rows) > max_rows
            if truncated:
                rows = rows[:max_rows]

            return {
                "columns": columns,
                "rows": [tuple(row) for row in rows],
                "truncated": truncated,
            }


# ── Scoped tool registry ─────────────────────────────────────────────

def scoped_registry(base_registry: ToolRegistry, scope: AgentScope) -> ToolRegistry:
    """Return a ``ToolRegistry`` that only exposes tools the scope allows.

    The returned registry shares the base registry's ``services``, ``config``,
    ``orchestrator``, and the tool instances themselves — it only filters
    the ``tools`` dict. When no tool filter is declared, the base registry
    is returned unchanged.
    """
    if not scope.has_tool_filter:
        return base_registry

    new_registry = ToolRegistry(base_registry.db, base_registry.config, base_registry.services)
    new_registry.orchestrator = base_registry.orchestrator

    allowed_names: set[str]
    if scope.tools_allow is not None:
        allowed_names = set(scope.tools_allow)
    else:
        allowed_names = {n for n in base_registry.tools.keys() if n not in (scope.tools_deny or set())}
    allowed_names = _expand_tool_dependencies(base_registry.tools, allowed_names)

    for name, tool in base_registry.tools.items():
        if name in allowed_names:
            new_registry.tools[name] = tool
    return new_registry


def scoped_db(base_db, scope: AgentScope):
    """Return a ``ScopedDatabase`` or the base db if no table filter is set."""
    if not scope.has_table_filter:
        return base_db
    return ScopedDatabase(base_db, scope)


def _is_safe_identifier(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _expand_tool_dependencies(tools: dict, names: set[str]) -> set[str]:
    expanded, pending = set(names), list(names)
    while pending:
        tool = tools.get(pending.pop())
        if tool is None:
            continue
        try:
            source = inspect.getsource(tool.__class__)
        except (OSError, TypeError):
            continue
        for dep in _CALL_TOOL_RE.findall(source):
            if dep in tools and dep not in expanded:
                expanded.add(dep)
                pending.append(dep)
    return expanded
