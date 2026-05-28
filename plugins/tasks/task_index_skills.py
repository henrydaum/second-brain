"""Path task that indexes canvas skill metadata into an FTS5 table.

Companion to ``embed_skills``: keyword (BM25) retrieval to complement the
dense embedding retriever. ``search_skills`` fuses the two with RRF so that
exact name/description matches surface reliably even when popularity or
embedding cosine would otherwise bury them.
"""

from __future__ import annotations

from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult
from plugins.skills.helpers.skill_meta import is_skill_module, read_skill_meta


class IndexSkills(BaseTask):
    name = "index_skills"
    modalities = ["text"]
    writes = ["skill_fts"]
    output_schema = """
        CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(
            slug, name, description, kind,
            path UNINDEXED,
            hidden UNINDEXED,
            tokenize = 'porter unicode61'
        );
    """
    batch_size = 16

    def run(self, paths: list[str], context) -> list[TaskResult]:
        db = context.db
        return [_index_path(path, db) for path in paths]


def _index_path(path: str, db) -> TaskResult:
    p = Path(path)
    if not is_skill_module(p):
        return TaskResult()
    try:
        meta = read_skill_meta(p)
        if meta is None:
            return TaskResult()
        # FTS5 has no primary key, so INSERT OR REPLACE won't upsert. Delete
        # any prior row for this path, then insert. Return data=[] so the
        # orchestrator doesn't double-write via write_outputs.
        with db.lock:
            db.conn.execute("DELETE FROM skill_fts WHERE path = ?", (str(p),))
            db.conn.execute(
                "INSERT INTO skill_fts (slug, name, description, kind, path, hidden) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (meta["slug"], meta["name"], meta["description"], meta["kind"], str(p), meta["hidden"]),
            )
            db.conn.commit()
        return TaskResult()
    except Exception as e:
        return TaskResult.failed(str(e))
