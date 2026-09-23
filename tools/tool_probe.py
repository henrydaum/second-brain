"""Ask typed Jev questions of the whole synced corpus.

The agent writes questions in Jev's own format (noul / score / choice) and a
target; this tool cuts the target into windows of consecutive chunks, checks
the questions with ``rlcd.validate``, refuses anything over the token cap, and
queues the work as ``probe.slice`` events for ``task_probe``. Results arrive
asynchronously in ``probe_answers``; the agent aggregates them with
``sql_query``. Statistics deliberately live there, not here — ``status`` gives
only the summary that tells a bad question from a good one.

Every submit is a new probe with its own id, so nothing is overwritten, and a
probe is a snapshot of the corpus at submit time.
"""

dependencies_files = ['services/service_rlcd.py', 'tasks/task_probe.py',
                      'tools/tool_sql_query.py']
dependencies_pip = []
requests = ["db.query", "db.write", "db.define", "service.call",
            "event.emit", "session.get", "config.read"]

import json
import time

from guest.bases import BaseTool

CHANNEL = "probe.slice"
WINDOW_CHARS = 4000     # ~1k tokens: far under Jev's 32k, since irrelevant state hurts
SLICE_UNITS = 50        # one task run; keeps a slice well inside the wall ceiling
INSERT_ROWS = 500       # rows per multi-row INSERT (db.write takes one statement)
PAGE = 500              # db.query's own row cap
DEFAULT_MAX_TOKENS = 2_000_000

PROBES_DDL = """CREATE TABLE IF NOT EXISTS probes (
    id INTEGER PRIMARY KEY, questions_json TEXT, state_context TEXT,
    target_json TEXT, window_chars INTEGER, session_key TEXT,
    created_at REAL, finished_at REAL)"""
UNITS_DDL = """CREATE TABLE IF NOT EXISTS probe_units (
    probe_id INTEGER, unit_id INTEGER, path TEXT, chunk_start INTEGER,
    chunk_end INTEGER, chars INTEGER, done_at REAL, error TEXT,
    PRIMARY KEY (probe_id, unit_id))"""


class Probe(BaseTool):
    """Corpus probes."""
    name = "probe"
    description = (
        "Ask typed yes/no, score and choice questions of every passage in the "
        "synced corpus, answered by the Jev model. Use it for questions that "
        "would need reading everything ('what is Henry's view of X', 'how often "
        "does Y come up and in what tone'). Asynchronous: submit returns a probe "
        "id; answers land in the probe_answers table for sql_query."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["submit", "status", "resume"]},
            "questions": {
                "type": "object",
                "description": "submit: Jev questions keyed by name, e.g. "
                               "{\"relevant\": {\"type\": \"noul\", \"instructions\": "
                               "\"Does `text` discuss Buddhism?\"}}.",
            },
            "target": {
                "type": "object",
                "description": "submit: which passages. {\"path_contains\": \"Journal\" "
                               "or [..], \"extensions\": [\".md\"]} (empty = whole "
                               "corpus), or {\"probe\": 12, \"question\": \"relevant\", "
                               "\"min\": 0.5} to re-ask the windows an earlier "
                               "probe answered.",
            },
            "state_context": {
                "type": "string",
                "description": "submit: one line every passage is read with, e.g. "
                               "'Personal writing by Henry.'",
            },
            "probe_id": {"type": "integer", "description": "status / resume."},
            "retry_errors": {"type": "boolean",
                             "description": "resume: also retry failed windows."},
        },
        "required": ["action"],
    }
    requires_services = ["rlcd"]
    config_settings = [
        ("Probe token cap", "probe_max_input_tokens",
         "Largest estimated Jev input (tokens) one probe may spend.",
         DEFAULT_MAX_TOKENS, {"type": "integer"}),
    ]
    agent_prompt = (
        "## Corpus probes (tool: probe)\n"
        "Jev answers typed questions about every passage; it never writes prose. "
        "Each passage is the state {\"file\", \"text\", \"context\"?}; refer to "
        "`text` in instructions. Jev reads literally: one plain judgement per "
        "question, no negations; never ask it about dates, counts or arithmetic — "
        "do those in SQL.\n"
        "Wide then deep: put a relevance noul beside the deep questions and weight "
        "by it, or submit a second probe with target {\"probe\": id, \"question\": "
        "\"relevant\", \"min\": 0.5} to ask only the relevant windows.\n"
        "Submit returns at once; a notification arrives when done, and status "
        "reports progress and which questions came back ambiguous. A probe is a "
        "snapshot: resubmit to cover new or changed files.\n"
        "probe_answers(probe_id, path, chunk_start, chunk_end, question, type, "
        "value, choice, confidence, probs_json): value is P(yes) for noul and the "
        "score for score. Use probabilities, not top answers, and aggregate in "
        "SQL (500-row cap):\n"
        "- expected count: SELECT SUM(value) FROM probe_answers WHERE probe_id=12 "
        "AND question='relevant'\n"
        "- weighted score: SELECT SUM(r.value*s.value)/SUM(r.value) FROM "
        "probe_answers r JOIN probe_answers s USING (probe_id, path, chunk_start) "
        "WHERE r.probe_id=12 AND r.question='relevant' AND s.question='stance'\n"
        "- choice counts: SELECT j.key, SUM(j.value) FROM probe_answers a, "
        "json_each(a.probs_json) j WHERE a.probe_id=12 AND a.question='aspect' "
        "GROUP BY j.key\n"
        "Passage text: JOIN text_chunks t ON t.path=a.path AND t.chunk_index "
        "BETWEEN a.chunk_start AND a.chunk_end. files.mtime is last-modified, not "
        "when something was written."
    )

    def run(self, sdk, action="", **kwargs):
        """Dispatch one action."""
        sdk.db.define(PROBES_DDL)
        sdk.db.define(UNITS_DDL)
        if action == "submit":
            return self._submit(sdk, kwargs.get("questions"),
                                kwargs.get("target") or {},
                                kwargs.get("state_context") or "")
        if action in ("status", "resume"):
            probe_id = kwargs.get("probe_id")
            if not isinstance(probe_id, int):
                return sdk.fail(f"{action} needs an integer probe_id.")
            if not sdk.db.query("SELECT 1 FROM probes WHERE id = ?", [probe_id]):
                return sdk.fail(f"Probe {probe_id} does not exist.")
            if action == "status":
                return self._status(sdk, probe_id)
            return self._resume(sdk, probe_id, bool(kwargs.get("retry_errors")))
        return sdk.fail("action must be submit, status or resume.")

    # ── submit ─────────────────────────────────────────────────────────

    def _submit(self, sdk, questions, target, context):
        if not isinstance(questions, dict) or not questions:
            return sdk.fail("submit needs questions: an object of Jev questions keyed by name.")
        if not isinstance(target, dict):
            return sdk.fail("target must be an object.")

        if "probe" in target:
            units = self._units_from_probe(sdk, target)
        else:
            units = self._units_from_paths(sdk, target)
        if isinstance(units, str):
            return sdk.fail(units)
        if not units:
            return sdk.fail("The target matches no passages. Check path_contains "
                            "against SELECT DISTINCT path FROM text_chunks.")

        placeholder = {"file": "x" * 40, "text": "x" * max(u["chars"] for u in units)}
        if context:
            placeholder["context"] = context
        verdict = sdk.services.call("rlcd", "validate", questions, state=placeholder)
        if not verdict.get("ok"):
            return sdk.fail(f"Questions not accepted: {verdict.get('reason')}")

        question_tokens = len(json.dumps(questions)) // 4 + 1
        estimate = sum(u["chars"] // 4 + question_tokens for u in units)
        cap = int(sdk.config.read("probe_max_input_tokens") or DEFAULT_MAX_TOKENS)
        if estimate > cap:
            by_file = {}
            for u in units:
                by_file[u["path"]] = by_file.get(u["path"], 0) + u["chars"]
            biggest = sorted(by_file, key=by_file.get, reverse=True)[:5]
            return sdk.fail(
                f"Estimated ~{estimate:,} input tokens over {len(units)} windows, "
                f"above probe_max_input_tokens ({cap:,}). Narrow the target, "
                f"shorten the questions, or raise the setting. Largest files: "
                + ", ".join(biggest))

        session = sdk.session.get() or {}
        probe_id = self._insert_probe(sdk, questions, context, target,
                                      session.get("key") or "")
        for start in range(0, len(units), INSERT_ROWS):
            batch = units[start:start + INSERT_ROWS]
            params = []
            for offset, u in enumerate(batch):
                params += [probe_id, start + offset, u["path"], u["chunk_start"],
                           u["chunk_end"], u["chars"]]
            sdk.db.write(
                "INSERT INTO probe_units (probe_id, unit_id, path, chunk_start, "
                "chunk_end, chars) VALUES " + ", ".join(["(?, ?, ?, ?, ?, ?)"] * len(batch)),
                params)
        slices = self._emit(sdk, probe_id, list(range(len(units))))
        files = len({u["path"] for u in units})
        return sdk.ok(
            {"probe_id": probe_id, "windows": len(units), "files": files,
             "estimated_tokens": estimate, "slices": slices},
            llm_summary=(f"Probe {probe_id} queued: {len(units)} windows across "
                         f"{files} files, ~{estimate:,} input tokens. Answers arrive "
                         f"asynchronously; a notification follows when it finishes. "
                         f"Check progress with action 'status'."))

    def _insert_probe(self, sdk, questions, context, target, session_key):
        """Insert the probe row and return its id (db.write returns no rowid)."""
        for _attempt in range(3):
            probe_id = (sdk.db.query("SELECT COALESCE(MAX(id), 0) + 1 AS n FROM probes")[0]["n"])
            try:
                sdk.db.write(
                    "INSERT INTO probes (id, questions_json, state_context, target_json, "
                    "window_chars, session_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [probe_id, json.dumps(questions), context, json.dumps(target),
                     WINDOW_CHARS, session_key, time.time()])
                return probe_id
            except sdk.Failed:
                continue  # another submit took the id
        raise RuntimeError("could not allocate a probe id")

    def _units_from_paths(self, sdk, target):
        """Windows of consecutive chunks over every file the target names."""
        contains = target.get("path_contains") or []
        if isinstance(contains, str):
            contains = [contains]
        extensions = [e.lower() if e.startswith(".") else "." + e.lower()
                      for e in (target.get("extensions") or [])]
        where, params = [], []
        if contains:
            where.append("(" + " OR ".join(["LOWER(path) LIKE ?"] * len(contains)) + ")")
            params += [f"%{c.lower()}%" for c in contains]
        if extensions:
            where.append("(" + " OR ".join(["LOWER(path) LIKE ?"] * len(extensions)) + ")")
            params += [f"%{e}" for e in extensions]
        clause = ("WHERE " + " AND ".join(where)) if where else ""

        units, current, offset = [], None, 0
        while True:
            rows = sdk.db.query(
                f"SELECT path, chunk_index, COALESCE(char_count, LENGTH(content)) AS n "
                f"FROM text_chunks {clause} ORDER BY path, chunk_index "
                f"LIMIT {PAGE} OFFSET {offset}", params)
            for row in rows:
                size = row["n"] or 0
                if (current is None or current["path"] != row["path"]
                        or current["chars"] + size > WINDOW_CHARS):
                    current = {"path": row["path"], "chunk_start": row["chunk_index"],
                               "chunk_end": row["chunk_index"], "chars": size}
                    units.append(current)
                else:
                    current["chunk_end"] = row["chunk_index"]
                    current["chars"] += size
            if len(rows) < PAGE:
                return units
            offset += PAGE

    def _units_from_probe(self, sdk, target):
        """The exact windows an earlier probe answered, filtered on one question."""
        try:
            source = int(target["probe"])
        except (KeyError, TypeError, ValueError):
            return "target.probe must be an earlier probe's integer id."
        question = target.get("question")
        if not question:
            return "target with probe also needs question (and min and/or max)."
        low = target.get("min", 0.0)
        high = target.get("max", 1e9)
        units, offset = [], 0
        while True:
            rows = sdk.db.query(
                "SELECT a.path, a.chunk_start, a.chunk_end, u.chars FROM probe_answers a "
                "JOIN probe_units u ON u.probe_id = a.probe_id AND u.path = a.path "
                "AND u.chunk_start = a.chunk_start "
                "WHERE a.probe_id = ? AND a.question = ? AND a.value BETWEEN ? AND ? "
                f"ORDER BY a.path, a.chunk_start LIMIT {PAGE} OFFSET {offset}",
                [source, question, low, high])
            units += [dict(r) for r in rows]
            if len(rows) < PAGE:
                break
            offset += PAGE
        if not units and not sdk.db.query(
                "SELECT 1 FROM probe_answers WHERE probe_id = ? AND question = ? LIMIT 1",
                [source, question]):
            return (f"Probe {source} has no answers for question '{question}' "
                    "(wrong name, or not finished yet).")
        return units

    def _emit(self, sdk, probe_id, unit_ids):
        """Queue contiguous runs of unit ids, at most SLICE_UNITS per event."""
        slices = 0
        runs, run = [], []
        for uid in unit_ids:
            if run and (uid != run[-1] + 1 or len(run) >= SLICE_UNITS):
                runs.append(run)
                run = []
            run.append(uid)
        if run:
            runs.append(run)
        for run in runs:
            sdk.events.emit(CHANNEL, {"probe_id": probe_id,
                                      "first_unit": run[0], "last_unit": run[-1]})
            slices += 1
        return slices

    # ── status / resume ────────────────────────────────────────────────

    def _status(self, sdk, probe_id):
        probe = sdk.db.query("SELECT * FROM probes WHERE id = ?", [probe_id])[0]
        counts = sdk.db.query(
            "SELECT COUNT(*) AS total, SUM(done_at IS NOT NULL) AS done, "
            "SUM(error IS NOT NULL) AS errored FROM probe_units WHERE probe_id = ?",
            [probe_id])[0]
        total, done = counts["total"] or 0, counts["done"] or 0
        errored = counts["errored"] or 0
        pending = total - done - errored
        lines = [f"Probe {probe_id}: {done}/{total} windows answered, "
                 f"{pending} pending, {errored} failed."
                 + (" Finished." if probe.get("finished_at") else "")]
        if errored:
            errors = sdk.db.query(
                "SELECT error, COUNT(*) AS n FROM probe_units WHERE probe_id = ? "
                "AND error IS NOT NULL GROUP BY error ORDER BY n DESC LIMIT 3", [probe_id])
            lines += [f"  failed x{e['n']}: {e['error']}" for e in errors]
            lines.append("  resume with retry_errors=true to try them again.")

        questions = json.loads(probe["questions_json"])
        summary = {}
        for name, q in questions.items():
            kind = q.get("type")
            row = sdk.db.query(
                "SELECT COUNT(*) AS n, AVG(value) AS mean, "
                "AVG(CASE WHEN type = 'noul' THEN value BETWEEN 0.2 AND 0.8 "
                "ELSE confidence < 0.5 END) AS uncertain "
                "FROM probe_answers WHERE probe_id = ? AND question = ?",
                [probe_id, name])[0]
            if not row["n"]:
                continue
            entry = {"answered": row["n"], "uncertain_share": round(row["uncertain"] or 0, 2)}
            if kind == "choice":
                top = sdk.db.query(
                    "SELECT j.key AS option, SUM(j.value) AS expected FROM probe_answers a, "
                    "json_each(a.probs_json) j WHERE a.probe_id = ? AND a.question = ? "
                    "GROUP BY j.key ORDER BY expected DESC LIMIT 3", [probe_id, name])
                entry["top"] = {t["option"]: round(t["expected"], 1) for t in top}
                shown = ", ".join(f"{k} {v}" for k, v in entry["top"].items())
            else:
                entry["mean"] = round(row["mean"], 3)
                shown = f"mean {entry['mean']}"
            summary[name] = entry
            line = (f"  {name} ({kind}): {row['n']} answered, {shown}, "
                    f"{entry['uncertain_share']:.0%} uncertain")
            if entry["uncertain_share"] > 0.5:
                line += (" — Jev reads literally; this question is probably "
                         "ambiguous. Reword it and submit a new probe.")
            lines.append(line)

        stale = sdk.db.query(
            "SELECT COUNT(DISTINCT u.path) AS n FROM probe_units u JOIN text_chunks t "
            "ON t.path = u.path WHERE u.probe_id = ? AND t.chunked_at > ?",
            [probe_id, probe["created_at"]])[0]["n"] or 0
        if stale:
            lines.append(f"  {stale} file(s) were re-chunked after this probe was "
                         "submitted; resubmit to answer their current text.")
        if probe.get("finished_at"):
            lines.append("Aggregate with sql_query over probe_answers (see the "
                         "probe guidance for the idioms).")
        return sdk.ok({"probe_id": probe_id, "total": total, "done": done,
                       "pending": pending, "errored": errored,
                       "finished": bool(probe.get("finished_at")),
                       "questions": summary, "stale_files": stale},
                      llm_summary="\n".join(lines))

    def _resume(self, sdk, probe_id, retry_errors):
        if retry_errors:
            sdk.db.write("UPDATE probe_units SET error = NULL "
                         "WHERE probe_id = ? AND done_at IS NULL", [probe_id])
            sdk.db.write("UPDATE probes SET finished_at = NULL WHERE id = ?", [probe_id])
        pending, offset = [], 0
        while True:
            rows = sdk.db.query(
                "SELECT unit_id FROM probe_units WHERE probe_id = ? AND done_at IS NULL "
                f"AND error IS NULL ORDER BY unit_id LIMIT {PAGE} OFFSET {offset}",
                [probe_id])
            pending += [r["unit_id"] for r in rows]
            if len(rows) < PAGE:
                break
            offset += PAGE
        if not pending:
            return sdk.ok({"probe_id": probe_id, "requeued": 0},
                          llm_summary=f"Probe {probe_id} has nothing pending.")
        slices = self._emit(sdk, probe_id, pending)
        return sdk.ok({"probe_id": probe_id, "requeued": len(pending), "slices": slices},
                      llm_summary=f"Probe {probe_id}: re-queued {len(pending)} "
                                  f"window(s) in {slices} slice(s).")
