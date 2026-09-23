"""Answer one slice of a corpus probe with Jev.

A probe is a set of typed Jev questions an agent wrote, aimed at part of the
synced corpus (``tool_probe`` submits it). The tool cuts the target into
*units* — one Jev call's worth of consecutive chunks from one file — and emits
``probe.slice`` once per ~50 of them. Each emit is a durable ``task_runs`` row,
so this task is the queue's consumer and nothing else: it holds no state, reads
the probe's questions from its row, answers the pending units in its range, and
writes one ``probe_answers`` row per (window, question).

Questions are data, not code. A new question set is a new probe row and a new
batch of events; nothing here is edited or re-registered to ask something new,
and a path trigger could not have done this at all — it runs once per file,
while every probe has to reach the same files again.

It writes its own rows rather than returning them, so a slice that stops
halfway (a rate limit, the wall ceiling) keeps what it answered, and the
``done_at`` stamps are exactly what ``tool_probe``'s resume re-queues against.
Retryable provider failures stop the slice and leave the rest pending; a
failure specific to one window is recorded on that unit and the slice goes on.
"""

dependencies_files = ['services/service_rlcd.py', 'tasks/task_chunk_text.py']
dependencies_pip = []
requests = ["db.query", "db.write", "service.call", "session.push",
            "event.emit", "self.budget"]

import json
import time

from guest.bases import BaseTask
from guest.parsing import basename

CHANNEL = "probe.slice"

# Stop starting new windows with this much wall clock left, and hand the rest
# back to the queue. A Jev call is seconds; the margin covers a slow one.
WALL_MARGIN = 45.0

# The longest overlap searched for when joining chunks. The chunker carries
# ~embed_chunk_overlap characters of whole segments forward, so a segment can
# push it past the setting; this only bounds the search.
MAX_OVERLAP = 2000

# Shorter matches are coincidence ("no" + "overlap"), not carried text: the
# chunker's overlap is whole segments, tens of characters at least.
MIN_OVERLAP = 20


def _join(chunks: list) -> str:
    """Concatenate consecutive chunks, dropping the overlap each one repeats.

    The chunker starts every chunk with the exact tail of the one before, so
    the overlap is the longest suffix of the text so far that the next chunk
    begins with. Found by trying lengths longest-first; ``endswith`` keeps each
    try cheap.
    """
    text = ""
    for chunk in chunks:
        chunk = chunk or ""
        if text:
            for size in range(min(len(text), len(chunk), MAX_OVERLAP), MIN_OVERLAP - 1, -1):
                if text.endswith(chunk[:size]):
                    chunk = chunk[size:]
                    break
        text += chunk
    return text


def _flatten(answer: dict) -> tuple:
    """(type, value, choice, confidence, probs_json) for one Jev answer."""
    kind = answer.get("type")
    if kind == "noul":
        return kind, answer.get("noul"), None, None, None
    probs = json.dumps(answer.get("probabilities") or {})
    if kind == "score":
        return kind, answer.get("score"), None, answer.get("confidence"), probs
    return kind, None, answer.get("choice"), answer.get("confidence"), probs


class Probe(BaseTask):
    """Answer a probe's questions over its pending units."""
    name = "probe_slice"
    description = (
        "Answer one slice of a corpus probe: evaluate the probe's Jev "
        "questions over each pending window and store the typed answers."
    )
    trigger = "event"
    trigger_channels = ["probe.slice"]  # literal: read by AST; equals CHANNEL
    reads = ["text_chunks"]
    writes = ["probe_answers"]
    output_schema = """CREATE TABLE IF NOT EXISTS probe_answers (
        probe_id    INTEGER,
        path        TEXT,
        chunk_start INTEGER,
        chunk_end   INTEGER,
        question    TEXT,
        type        TEXT,
        value       REAL,
        choice      TEXT,
        confidence  REAL,
        probs_json  TEXT,
        model       TEXT,
        PRIMARY KEY (probe_id, path, chunk_start, question)
    )"""
    requires_services = ["rlcd"]
    timeout = 600
    # The rlcd service is one resident box and serializes its calls, so a
    # second worker would only queue behind the first.
    max_workers = 1

    def run_event(self, sdk, payload):
        """Answer the pending units between first_unit and last_unit."""
        try:
            probe_id = int(payload["probe_id"])
            first, last = int(payload["first_unit"]), int(payload["last_unit"])
        except (KeyError, TypeError, ValueError):
            return sdk.fail(f"probe.slice payload needs probe_id, first_unit, "
                            f"last_unit; got {payload!r}")

        rows = sdk.db.query(
            "SELECT questions_json, state_context FROM probes WHERE id = ?",
            [probe_id])
        if not rows:
            return sdk.fail(f"probe {probe_id} does not exist")
        questions = json.loads(rows[0]["questions_json"])
        context = rows[0].get("state_context") or ""

        units = sdk.db.query(
            "SELECT unit_id, path, chunk_start, chunk_end FROM probe_units "
            "WHERE probe_id = ? AND unit_id BETWEEN ? AND ? "
            "AND done_at IS NULL AND error IS NULL ORDER BY unit_id",
            [probe_id, first, last])

        answered = 0
        for index, unit in enumerate(units):
            wall = (sdk.budget() or {}).get("wall")
            if wall is not None and wall < WALL_MARGIN:
                rest = units[index:]
                sdk.events.emit(CHANNEL, {"probe_id": probe_id,
                                          "first_unit": rest[0]["unit_id"],
                                          "last_unit": rest[-1]["unit_id"]})
                sdk.log(f"probe {probe_id}: out of time, re-queued "
                        f"{len(rest)} unit(s)")
                break
            stop = self._answer(sdk, probe_id, unit, questions, context)
            if stop:
                return stop
            answered += 1

        self._finish_if_done(sdk, probe_id)
        return sdk.ok({"probe_id": probe_id, "answered": answered},
                      llm_summary=f"probe {probe_id}: answered {answered} window(s)")

    def _answer(self, sdk, probe_id, unit, questions, context):
        """Answer one unit. Returns a failure Result to stop the slice, or None."""
        chunks = sdk.db.query(
            "SELECT content FROM text_chunks WHERE path = ? "
            "AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
            [unit["path"], unit["chunk_start"], unit["chunk_end"]])
        if not chunks:
            self._mark_error(sdk, probe_id, unit,
                             "chunks no longer exist (file removed or re-chunked)")
            return None

        state = {"file": basename(unit["path"]),
                 "text": _join([row["content"] for row in chunks])}
        if context:
            state["context"] = context

        try:
            result = sdk.services.call("rlcd", "evaluate", state, questions)
        except sdk.Failed as failed:
            # evaluate raises only on configuration or invalid input, which
            # every remaining window would hit the same way.
            return sdk.fail(f"probe {probe_id}: rlcd refused the call: {failed}")

        if not result.get("ok"):
            error = result.get("error") or {}
            if error.get("retryable") or error.get("kind") == "transport":
                # Rate limits, provider outages and a blocked host are about
                # the environment, not this window: stop, leave it pending.
                after = error.get("retry_after")
                return sdk.fail(
                    f"probe {probe_id}: provider unavailable "
                    f"({error.get('kind')} {error.get('status', '')}); "
                    f"remaining units stay pending — resume the probe later"
                    + (f" (retry after {after}s)" if after else ""),
                    retryable=True)
            self._mark_error(sdk, probe_id, unit,
                             f"{error.get('kind')}: {error.get('message')}")
            return None

        values, params = [], []
        for question, answer in result["answers"].items():
            kind, value, choice, confidence, probs = _flatten(answer)
            values.append("(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
            params += [probe_id, unit["path"], unit["chunk_start"],
                       unit["chunk_end"], question, kind, value, choice,
                       confidence, probs, result.get("model")]
        sdk.db.write(
            "INSERT OR REPLACE INTO probe_answers (probe_id, path, chunk_start, "
            "chunk_end, question, type, value, choice, confidence, probs_json, "
            "model) VALUES " + ", ".join(values), params)
        sdk.db.write(
            "UPDATE probe_units SET done_at = ? WHERE probe_id = ? AND unit_id = ?",
            [time.time(), probe_id, unit["unit_id"]])
        return None

    def _mark_error(self, sdk, probe_id, unit, message):
        """Record a failure specific to one window, so the slice can go on."""
        sdk.db.write(
            "UPDATE probe_units SET error = ? WHERE probe_id = ? AND unit_id = ?",
            [str(message)[:500], probe_id, unit["unit_id"]])

    def _finish_if_done(self, sdk, probe_id):
        """Stamp the probe finished and notify, once, when nothing is pending."""
        counts = sdk.db.query(
            "SELECT SUM(done_at IS NULL AND error IS NULL) AS pending, "
            "SUM(done_at IS NOT NULL) AS done, SUM(error IS NOT NULL) AS errored "
            "FROM probe_units WHERE probe_id = ?", [probe_id])[0]
        if counts.get("pending"):
            return
        stamp = time.time()
        sdk.db.write("UPDATE probes SET finished_at = ? "
                     "WHERE id = ? AND finished_at IS NULL", [stamp, probe_id])
        row = sdk.db.query("SELECT finished_at, session_key FROM probes "
                           "WHERE id = ?", [probe_id])[0]
        if row.get("finished_at") != stamp:
            return  # another run finished it first and has already said so
        errored = counts.get("errored") or 0
        message = (f"Probe {probe_id} answered {counts.get('done') or 0} window(s)"
                   + (f"; {errored} failed" if errored else "")
                   + f". Call the probe tool with action 'status' and "
                     f"probe_id {probe_id} for a summary, then query "
                     f"probe_answers with sql_query.")
        try:
            sdk.session.push(message, key=row.get("session_key") or "",
                             title=f"Probe {probe_id} finished", notify=True,
                             level="warning" if errored else "success")
        except sdk.Failed as failed:
            sdk.log(f"probe {probe_id} finished but could not notify: {failed}")
