"""A migrated task has two entry points, and only one used to be bridged.

An event-triggered task declares ``trigger = "event"`` and writes
``run_event``; the orchestrator calls that, not ``run``. Nothing forwarded it,
so such a task loaded, registered, subscribed, and then answered every firing
with the native base class's do-nothing default — recorded as a *successful*
run. Silence shaped like success is the worst way for a gap to present, so both
halves are pinned here: the doorway exists, and the declaration that routes to
it survives being read.
"""

from types import SimpleNamespace

import pytest

from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.validator import validate_file

_EVENT_TASK = '''\
"""A task that reacts to a channel."""

from guest.bases import BaseTask

requests = []


class SweepTask(BaseTask):
    """Sweep."""

    name = "sweep"
    description = "x"
    trigger = "event"
    trigger_channels = ["sweep_now"]

    def run_event(self, sdk, payload):
        """Answer with what arrived."""
        return sdk.ok({"seen": payload.get("mark")})
'''

_PATH_TASK = '''\
"""A task that reacts to files."""

from guest.bases import BaseTask

requests = []


class IndexTask(BaseTask):
    """Index."""

    name = "index"
    description = "x"

    def run(self, sdk, paths):
        """Answer with what arrived."""
        return sdk.ok({"count": len(paths)})
'''


@pytest.fixture(autouse=True)
def _sandbox():
    configure(Sandbox())


def _adapted(tmp_path, source, stem):
    tasks = tmp_path / "tasks"
    tasks.mkdir(exist_ok=True)
    path = tasks / f"task_{stem}.py"
    path.write_text(source, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the file should have adapted as a task"
    return module


def _instance(module):
    name = next(n for n in dir(module) if n.startswith("Sandboxed"))
    return getattr(module, name)()


def test_an_event_task_reaches_its_guest(tmp_path):
    """The orchestrator's call signature, forwarded to the guest's."""
    task = _instance(_adapted(tmp_path, _EVENT_TASK, "sweep"))

    result = task.run_event("run-1", {"mark": 7}, SimpleNamespace())

    assert result.success
    assert result.data == {"seen": 7}


def test_an_event_task_answers_with_a_task_result(tmp_path):
    """``run_event`` is an entry point, so it gets the family's translation.

    Handing the orchestrator raw data instead would make a failed sweep
    indistinguishable from a successful one.
    """
    task = _instance(_adapted(tmp_path, _EVENT_TASK, "sweep"))
    result = task.run_event("run-1", {}, SimpleNamespace())

    assert hasattr(result, "success") and hasattr(result, "error")


def test_a_path_task_does_not_grow_the_doorway(tmp_path):
    """Carried only when the guest defines one, like ``form`` on a command.

    An adapter advertising ``run_event`` it cannot fulfil would answer the
    orchestrator by forwarding into nothing.
    """
    module = _adapted(tmp_path, _PATH_TASK, "index")
    task = _instance(module)

    assert "run_event" not in vars(type(task))
    assert task.run(["a", "b"], SimpleNamespace()).data == {"count": 2}


def test_the_channel_declaration_survives_being_read(tmp_path):
    """Declarations are AST-read, so a *name* reads as nothing at all.

    ``trigger_channels = [CHANNEL]`` is the natural way to write it and used to
    produce a task subscribed to no channel: it validated, loaded, registered,
    and never fired. The validator now refuses it at authoring time.
    """
    task = _instance(_adapted(tmp_path, _EVENT_TASK, "sweep"))
    assert task.trigger_channels == ["sweep_now"]


def test_a_channel_named_by_reference_is_refused(tmp_path):
    """The failure this rule exists for, pinned as a refusal."""
    source = _EVENT_TASK.replace(
        'trigger_channels = ["sweep_now"]',
        'CHANNEL = "sweep_now"\n    trigger_channels = [CHANNEL]')
    path = tmp_path / "task_indirect.py"
    path.write_text(source, encoding="utf-8")

    report = validate_file(path)

    assert not report.ok
    assert "trigger_channels" in report.render()
