"""
Agent scope — per-profile lens over the shared database and tool registry.

An agent profile (stored in ``agent_profiles`` config) declares whitelist or
blacklist filters for the tables it can read and the tools it can call.
This module turns those declarations into:

    - a ``ScopedDatabase`` that runs queries against an in-memory main
      schema with TEMP VIEWs redirecting to the real DB (attached read-only
      as ``source``), plus a SQLite authorizer that denies top-level reads
      of the ``source`` schema so scoped queries can only reach data
      through the views we create;
    - a ``ToolRegistry`` view that only exposes the allowed tools.

Using ``blacklist`` with an empty list means no restriction.
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
    tables_allow: list | None = None  # whitelist entries: str | dict{name, sql}
    tables_deny: set[str] | None = None
    folders_allow: list[str] | None = None  # normalized fwd-slash, no trailing slash
    folders_deny: list[str] | None = None

    @property
    def has_tool_filter(self) -> bool:
        return self.tools_allow is not None or bool(self.tools_deny)

    @property
    def has_table_filter(self) -> bool:
        return self.tables_allow is not None or bool(self.tables_deny)

    @property
    def has_folder_filter(self) -> bool:
        # An explicit (even empty) whitelist counts — empty whitelist = fail closed.
        return self.folders_allow is not None or bool(self.folders_deny)


def load_scope(profile_name: str, config: dict) -> AgentScope:
    """Parse a profile's filter fields into an ``AgentScope``."""
    profile = config.get("agent_profiles", {}).get(profile_name, {}) or {}

    tools_mode = _scope_mode(profile_name, profile, "tools")
    tables_mode = _scope_mode(profile_name, profile, "tables")
    folders_mode = _scope_mode(profile_name, profile, "folders")
    tools_list = _scope_list(profile_name, profile, "tools")
    tables_list = _scope_list(profile_name, profile, "tables")
    folders_list = _scope_list(profile_name, profile, "folders")

    folders_norm = _normalize_folders(profile_name, folders_list)

    return AgentScope(
        profile_name=profile_name,
        prompt_suffix=str(profile.get("prompt_suffix") or ""),
        tools_allow=set(tools_list) if tools_mode == "whitelist" else None,
        tools_deny=set(tools_list) if tools_mode == "blacklist" else None,
        tables_allow=list(tables_list) if tables_mode == "whitelist" else None,
        tables_deny=set(tables_list) if tables_mode == "blacklist" else None,
        folders_allow=folders_norm if folders_mode == "whitelist" else None,
        folders_deny=folders_norm if folders_mode == "blacklist" else None,
    )


def _normalize_folders(profile_name: str, raw: list) -> list[str]:
    """Coerce folder strings to forward-slash, no-trailing-slash form, dedup, drop empties."""
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError(
                f"Profile '{profile_name}' folders_list entry must be a string, got {type(entry).__name__}"
            )
        norm = entry.replace("\\", "/").rstrip("/")
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _scope_mode(profile_name: str, profile: dict, kind: str) -> str:
    key = f"whitelist_or_blacklist_{kind}"
    mode = profile.get(key, "blacklist")
    if mode not in ("whitelist", "blacklist"):
        raise ValueError(
            f"Profile '{profile_name}' has invalid {key}: {mode!r}. "
            "Use 'whitelist' or 'blacklist'."
        )
    return mode


def _scope_list(profile_name: str, profile: dict, kind: str) -> list:
    key = f"{kind}_list"
    value = profile.get(key, [])
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"Profile '{profile_name}' field {key} must be an array.")
    return value


# ── Scoped database ──────────────────────────────────────────────────

class ScopedDatabase:
    """Read-only lens over a ``Database``.

    The connection has an empty in-memory main schema with only the views
    the profile is allowed to see; the real DB is attached read-only as
    ``source``. Named-view SQL in ``tables_list`` must reference base
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
        if not (scope.has_table_filter or scope.has_folder_filter):
            raise ValueError("ScopedDatabase should only be built when tables or folders are restricted.")
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
        self._folder_clause = _folder_where_clause(scope)
        self._has_path_col_cache: dict[str, bool] = {}
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
                        logger.warning(f"tables_list references missing object: {entry!r}")
                        continue
                    if not _is_safe_identifier(entry):
                        logger.warning(f"tables_list bare name is not a safe identifier: {entry!r}")
                        continue
                    # Passthrough view in main -> source.<name>, with optional folder filter.
                    self._create_view(entry, self._passthrough_sql(entry))
                elif isinstance(entry, dict):
                    name = entry.get("name")
                    sql = entry.get("sql")
                    if not name or not sql:
                        logger.warning(f"tables_list entry missing name/sql: {entry!r}")
                        continue
                    if not _is_safe_identifier(name):
                        logger.warning(f"tables_list entry has invalid name: {name!r}")
                        continue
                    # Custom-SQL entries are user-authored — folder filter is NOT auto-applied.
                    self._create_view(name, sql)
                else:
                    logger.warning(f"Unrecognized tables_list entry: {entry!r}")
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
                self._create_view(name, self._passthrough_sql(name))

    def _passthrough_sql(self, name: str) -> str:
        """Build the SELECT for a passthrough view, applying the folder filter if applicable."""
        base = f"SELECT * FROM {_SOURCE_SCHEMA}.{name}"
        if self._folder_clause and self._table_has_path_column(name):
            return f"{base} WHERE {self._folder_clause}"
        return base

    def _table_has_path_column(self, name: str) -> bool:
        cached = self._has_path_col_cache.get(name)
        if cached is not None:
            return cached
        try:
            cur = self._conn.execute(f"PRAGMA {_SOURCE_SCHEMA}.table_info({name})")
            has_path = any(row["name"] == "path" for row in cur.fetchall())
        except sqlite3.Error as e:
            logger.warning(f"PRAGMA table_info failed for {name!r}: {e}")
            has_path = False
        self._has_path_col_cache[name] = has_path
        return has_path

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
            # COUNT(*) and similar top-level table reads arrive without a
            # schema/column name, so deny them unless they target one of the
            # scoped main-schema views we created.
            if not db_name and not arg2 and arg1 not in self._visible_names:
                return sqlite3.SQLITE_DENY
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

def scoped_registry(base_registry: ToolRegistry, scope: AgentScope, db=None) -> ToolRegistry:
    """Return a ``ToolRegistry`` that only exposes tools the scope allows.

    The returned registry shares the base registry's ``services``, ``config``,
    ``orchestrator``, and the tool instances themselves. It may filter the
    ``tools`` dict and/or swap in a scoped database handle. When neither the
    tool set nor database changes, the base registry is returned unchanged.
    """
    target_db = base_registry.db if db is None else db
    if not scope.has_tool_filter and target_db is base_registry.db:
        return base_registry

    new_registry = ToolRegistry(target_db, base_registry.config, base_registry.services)
    new_registry.orchestrator = base_registry.orchestrator

    allowed_names: set[str]
    if not scope.has_tool_filter:
        allowed_names = set(base_registry.tools.keys())
    elif scope.tools_allow is not None:
        allowed_names = set(scope.tools_allow)
    else:
        allowed_names = {n for n in base_registry.tools.keys() if n not in (scope.tools_deny or set())}
    allowed_names = _expand_tool_dependencies(base_registry.tools, allowed_names)

    for name, tool in base_registry.tools.items():
        if name in allowed_names:
            new_registry.tools[name] = tool
    return new_registry


def resolve_agent_llm(profile_name: str, config: dict, services: dict):
    """Resolve the LLM service an agent profile should use.

    Looks up ``agent_profiles[profile_name]["llm"]``. The literal string
    ``"default"`` resolves to ``config["default_llm_profile"]``. Returns the
    service instance from ``services`` or ``None`` if it's missing — callers
    should fall back to ``services["llm"]`` (the default-LLM router) in that
    case so the app still works during transient misconfiguration.
    """
    profile = (config.get("agent_profiles", {}) or {}).get(profile_name, {}) or {}
    llm_ref = profile.get("llm") or "default"
    if llm_ref == "default":
        llm_ref = config.get("default_llm_profile") or ""
    if not llm_ref:
        return services.get("llm")
    return services.get(llm_ref) or services.get("llm")


def scoped_db(base_db, scope: AgentScope):
    """Return a ``ScopedDatabase`` or the base db if no scope filters are set."""
    if not (scope.has_table_filter or scope.has_folder_filter):
        return base_db
    return ScopedDatabase(base_db, scope)


def _folder_where_clause(scope: AgentScope) -> str | None:
    """Build a SQL predicate (no leading WHERE) restricting `path` to the scope's folders.

    Whitelist + non-empty folders → predicate matches paths inside any folder.
    Whitelist + empty folders     → '1=0' (fail closed).
    Blacklist + non-empty folders → NOT (predicate).
    No folder filter active       → None.

    Both sides of the comparison are normalized to forward slashes so that
    paths stored with backslashes (Windows) match folder strings normalized
    by ``_normalize_folders``. View bodies are static SQL, so folder strings
    are inlined with single quotes doubled — no other characters need escaping
    because LIKE wildcards (`%`, `_`) inside a folder name would only narrow
    the match (they can't escape the LIKE pattern boundary).
    """
    if scope.folders_allow is not None:
        folders = scope.folders_allow
        if not folders:
            return "1=0"
        predicate = _folder_predicate(folders)
        return predicate
    if scope.folders_deny:
        return f"NOT {_folder_predicate(scope.folders_deny)}"
    return None


def _folder_predicate(folders: list[str]) -> str:
    terms = []
    for folder in folders:
        escaped = folder.replace("'", "''")
        terms.append(
            f"(replace(path,'\\','/') = '{escaped}' "
            f"OR replace(path,'\\','/') LIKE '{escaped}/%')"
        )
    return "(" + " OR ".join(terms) + ")"


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
