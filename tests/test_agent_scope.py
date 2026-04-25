import unittest
from pathlib import Path

from agent.tool_registry import ToolRegistry
from frontend.commands import _normalize_agent_profile
from pipeline.database import Database
from plugins.tools.tool_sql_query import SQLQuery
from runtime.agent_scope import load_scope, scoped_db, scoped_registry


class AgentScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_tmp"
        self.tmp.mkdir(exist_ok=True)
        self.db_path = self.tmp / f"{self._testMethodName}.db"
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                path.unlink()
        self.db = Database(self.db_path)
        self.scoped = None
        conv_id = self.db.create_conversation("Scoped")
        self.db.save_message(conv_id, "user", "hello")
        self.registry = ToolRegistry(self.db, {}, {})
        self.registry.register(SQLQuery())

    def tearDown(self):
        if self.scoped is not None:
            self.scoped.conn.close()
        self.db.conn.close()
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                path.unlink()

    def test_scoped_registry_uses_scoped_db_for_tool_calls(self):
        config = {"agent_profiles": {"communicator": {"tables_allow": ["conversations", "conversation_messages"]}}}
        scope = load_scope("communicator", config)
        self.scoped = scoped_db(self.db, scope)
        registry = scoped_registry(self.registry, scope, db=self.scoped)
        ok = registry.call("sql_query", sql="SELECT id, title FROM conversations")
        blocked = registry.call("sql_query", sql="SELECT COUNT(*) AS n FROM files")
        self.assertTrue(ok.success)
        self.assertFalse(blocked.success)
        self.assertIn("Scoped", ok.llm_summary)
        self.assertIn("Access denied by agent scope", blocked.error)

    def test_missing_and_null_scope_keys_match(self):
        missing = load_scope("p", {"agent_profiles": {"p": {"llm": "default"}}})
        nulls = load_scope("p", {"agent_profiles": {"p": {"llm": "default", "tools_allow": None, "tools_deny": None, "tables_allow": None, "tables_deny": None}}})
        self.assertEqual(missing, nulls)

    def test_agent_profile_normalization_fills_missing_scope_keys(self):
        profile = _normalize_agent_profile({"llm": "default", "tables_allow": ["conversations"]})
        self.assertEqual(profile["tables_allow"], ["conversations"])
        self.assertIsNone(profile["tools_allow"])
        self.assertIsNone(profile["tools_deny"])
        self.assertIsNone(profile["tables_deny"])

    def test_unfiltered_scope_keeps_full_registry(self):
        scope = load_scope("default", {"agent_profiles": {"default": {"llm": "default"}}})
        self.assertIs(scoped_registry(self.registry, scope), self.registry)
        self.assertTrue(self.registry.call("sql_query", sql="SELECT COUNT(*) AS n FROM files").success)


if __name__ == "__main__":
    unittest.main()
