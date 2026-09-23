"""Corpus probes: tool_probe submits, task_probe answers, service_rlcd validates.

Store code, so marked ``store``. The three files run against a fake SDK whose
database is a real in-memory sqlite and whose ``services.call`` reaches a real
``RLCD`` instance — only the HTTP hop is synthetic, answering with schema-valid
System One responses so the service's own response validation runs too.
"""

import json
import sqlite3

import pytest

import sandbox  # noqa: F401  — aliases the guest package as ``guest``
from tests.support import store_source

pytestmark = pytest.mark.store


def _load(relative):
    source = store_source(relative)
    if source is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    namespace = {"__name__": f"test_{relative.rsplit('/', 1)[-1][:-3]}"}
    exec(compile(source, relative, "exec"), namespace)
    return namespace


class Failed(Exception):
    pass


class _Result(dict):
    pass


class _DB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

    def query(self, sql, params=None, max_rows=0):
        return [dict(r) for r in self.conn.execute(sql, params or [])][:500]

    def write(self, sql, params=None):
        try:
            self.conn.execute(sql, params or [])
        except sqlite3.Error as error:
            raise Failed(str(error))
        return True

    def define(self, ddl):
        self.conn.execute(ddl)
        return True


class _HTTP:
    """Synthetic System One: a fixed, valid answer per question type."""

    def __init__(self):
        self.status = 200
        self.calls = 0

    def http(self, url, method="GET", headers=None, json=None):
        self.calls += 1
        if self.status != 200:
            return {"status": self.status, "headers": {"Retry-After": "7"}, "body": ""}
        answers = {}
        for name, q in json["questions"].items():
            if q["type"] == "noul":
                p = 0.9 if "Buddhism" in json["state"]["text"] else 0.1
                answers[name] = {"type": "noul", "noul": p}
            elif q["type"] == "score":
                n = len(q["criteria"])
                probs = {str(i): (1.0 if i == n - 1 else 0.0) for i in range(n)}
                answers[name] = {"type": "score", "score": float(n - 1), "confidence": 0.9,
                                 "probabilities": probs,
                                 "legend": {str(i): c for i, c in enumerate(q["criteria"])}}
            else:
                keys = list(q["criteria"])
                probs = {k: (0.7 if i == 0 else 0.3 / max(len(keys) - 1, 1))
                         for i, k in enumerate(keys)}
                answers[name] = {"type": "choice", "choice": keys[0], "confidence": 0.4,
                                 "probabilities": probs}
        body = {"model": "jev-test", "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 0}}
        import json as _json
        return {"status": 200, "headers": {}, "body": _json.dumps(body)}


class FakeSDK:
    Failed = Failed
    Denied = Failed

    def __init__(self, db, rlcd, http, config=None):
        self.db = db
        self._rlcd, self._http = rlcd, http
        self._config = {"rlcd_endpoint": "http://127.0.0.1:9/v1/systemone",
                        "rlcd_model": "jev-test", **(config or {})}
        self.emitted, self.pushed = [], []
        outer = self

        class _Events:
            def emit(self, channel, payload=None):
                outer.emitted.append((channel, payload))
                return True

        class _Session:
            def get(self, key=""):
                return {"key": "session-1"}

            def push(self, message, key="", **kwargs):
                outer.pushed.append((key, message, kwargs))

        class _Config:
            def read(self, key):
                return outer._config.get(key)

        class _Services:
            def call(self, service, method, *args, **kwargs):
                try:
                    return getattr(outer._rlcd, method)(outer, *args, **kwargs)
                except ValueError as error:
                    raise Failed(str(error))

        class _Net:
            def http(self, *args, **kwargs):
                return outer._http.http(*args, **kwargs)

        self.events, self.session = _Events(), _Session()
        self.config, self.services, self.net = _Config(), _Services(), _Net()

    def ok(self, data=None, llm_summary="", **_):
        return _Result(ok=True, data=data, summary=llm_summary)

    def fail(self, error, retryable=False):
        return _Result(ok=False, error=error, retryable=retryable)

    def budget(self):
        return {"wall": 500.0, "running": 500.0}

    def log(self, *args, **kwargs):
        pass


QUESTIONS = {
    "relevant": {"type": "noul", "instructions": "Does `text` discuss Buddhism?"},
    "stance": {"type": "score", "instructions": "Stance toward it",
               "criteria": ["Critical", "Neutral", "Embracing"]},
    "aspect": {"type": "choice", "instructions": "Which aspect?",
               "criteria": {"meditation": None, "ethics": None, "other": None}},
}


@pytest.fixture
def world():
    service = _load("services/service_rlcd.py")
    task = _load("tasks/task_probe.py")
    tool = _load("tools/tool_probe.py")
    db = _DB()
    db.define("CREATE TABLE text_chunks (path TEXT, chunk_index INTEGER, content TEXT, "
              "char_count INTEGER, chunked_at REAL, PRIMARY KEY (path, chunk_index))")
    db.define(task["Probe"].output_schema)
    # Three files: two small, one big enough to span several windows.
    rows = [("C:/w/zen.md", 0, "On Buddhism and sitting.", 1.0),
            ("C:/w/zen.md", 1, "More on Buddhism.", 1.0),
            ("C:/w/tax.md", 0, "Quarterly taxes.", 1.0)]
    rows += [("C:/w/long.md", i, "y" * 1500, 1.0) for i in range(8)]
    for path, index, content, at in rows:
        db.write("INSERT INTO text_chunks VALUES (?, ?, ?, ?, ?)",
                 [path, index, content, len(content), at])
    http = _HTTP()
    sdk = FakeSDK(db, service["RLCD"](), http)
    return {"sdk": sdk, "db": db, "http": http,
            "tool": tool["Probe"](), "task": task["Probe"](), "task_ns": task}


def _drain(world):
    """Feed every emitted slice to the task, as the orchestrator would."""
    results = []
    while world["sdk"].emitted:
        _channel, payload = world["sdk"].emitted.pop(0)
        results.append(world["task"].run_event(world["sdk"], payload))
    return results


def test_validate_rejects_malformed_and_oversized_questions(world):
    rlcd, sdk = world["sdk"]._rlcd, world["sdk"]
    assert rlcd.validate(sdk, QUESTIONS) == {"ok": True}
    bad = rlcd.validate(sdk, {"q": {"type": "maybe", "instructions": "?"}})
    assert not bad["ok"] and "type" in bad["reason"]
    huge = rlcd.validate(sdk, QUESTIONS, state="x" * 140_000)
    assert not huge["ok"] and "limit" in huge["reason"]


def test_windows_respect_the_window_size(world):
    units = world["tool"]._units_from_paths(world["sdk"], {})
    assert all(u["chars"] <= 4000 or u["chunk_start"] == u["chunk_end"] for u in units)
    long = [u for u in units if u["path"].endswith("long.md")]
    assert len(long) == 4                     # 8 x 1500 chars, two per window
    assert [(u["chunk_start"], u["chunk_end"]) for u in long] == [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_submit_answers_and_finishes_once(world):
    result = world["tool"].run(world["sdk"], action="submit", questions=QUESTIONS,
                               target={}, state_context="Personal writing by Henry.")
    assert result["ok"], result
    probe_id = result["data"]["probe_id"]
    assert all(r["ok"] for r in _drain(world))

    rows = world["db"].query("SELECT * FROM probe_answers WHERE probe_id = ?", [probe_id])
    windows = result["data"]["windows"]
    assert len(rows) == windows * len(QUESTIONS)
    zen = {r["question"]: r for r in rows if r["path"].endswith("zen.md")}
    assert zen["relevant"]["value"] == pytest.approx(0.9)
    assert zen["stance"]["value"] == 2.0 and zen["stance"]["probs_json"]
    assert zen["aspect"]["choice"] == "meditation" and zen["aspect"]["value"] is None
    expected = world["db"].query(
        "SELECT SUM(j.value) AS n FROM probe_answers a, json_each(a.probs_json) j "
        "WHERE a.probe_id = ? AND a.question = 'aspect' AND j.key = 'meditation'",
        [probe_id])[0]["n"]
    assert expected == pytest.approx(0.7 * windows)

    assert len(world["sdk"].pushed) == 1
    assert world["sdk"].pushed[0][0] == "session-1"
    status = world["tool"].run(world["sdk"], action="status", probe_id=probe_id)
    assert status["data"]["finished"] and status["data"]["pending"] == 0
    assert "ambiguous" in status["summary"]   # choice confidence 0.4 on every row


def test_overlap_is_removed_when_joining(world):
    join = world["task_ns"]["_join"]
    carried = "the tail the chunker carries forward. "
    assert join(["Start of it, " + carried, carried + "and the rest."]) ==         "Start of it, " + carried + "and the rest."
    assert join(["no", "overlap"]) == "nooverlap"


def test_token_cap_refuses_a_submit(world):
    world["sdk"]._config["probe_max_input_tokens"] = 100
    result = world["tool"].run(world["sdk"], action="submit", questions=QUESTIONS, target={})
    assert not result["ok"] and "probe_max_input_tokens" in result["error"]
    assert not world["sdk"].emitted


def test_rate_limit_leaves_units_pending_and_resume_requeues_them(world):
    tool, sdk = world["tool"], world["sdk"]
    probe_id = tool.run(sdk, action="submit", questions=QUESTIONS, target={})["data"]["probe_id"]
    world["http"].status = 429
    results = _drain(world)
    assert results and not results[0]["ok"] and results[0]["retryable"]
    pending = world["db"].query("SELECT COUNT(*) AS n FROM probe_units WHERE probe_id = ? "
                                "AND done_at IS NULL AND error IS NULL", [probe_id])[0]["n"]
    assert pending > 0 and not sdk.pushed

    world["http"].status = 200
    resumed = tool.run(sdk, action="resume", probe_id=probe_id)
    assert resumed["data"]["requeued"] == pending
    _drain(world)
    assert tool.run(sdk, action="status", probe_id=probe_id)["data"]["finished"]


def test_a_second_probe_can_target_the_first_ones_windows(world):
    tool, sdk = world["tool"], world["sdk"]
    first = tool.run(sdk, action="submit", questions=QUESTIONS, target={})["data"]["probe_id"]
    _drain(world)
    deep = {"why": {"type": "noul", "instructions": "Is this Henry's own view?"}}
    second = tool.run(sdk, action="submit", questions=deep,
                      target={"probe": first, "question": "relevant", "min": 0.5})
    assert second["ok"], second
    units = world["db"].query("SELECT path, chunk_start, chunk_end FROM probe_units "
                              "WHERE probe_id = ?", [second["data"]["probe_id"]])
    assert [u["path"] for u in units] == ["C:/w/zen.md"]
    _drain(world)
    first_rows = world["db"].query("SELECT COUNT(*) AS n FROM probe_answers WHERE probe_id = ?",
                                   [first])[0]["n"]
    assert first_rows > 0                      # nothing overwritten


def test_invalid_questions_are_refused_before_anything_is_queued(world):
    result = world["tool"].run(world["sdk"], action="submit",
                               questions={"q": {"type": "score", "instructions": "x",
                                                "criteria": ["only one"]}}, target={})
    assert not result["ok"] and "Questions not accepted" in result["error"]
    assert not world["sdk"].emitted
