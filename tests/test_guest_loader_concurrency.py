"""One box member, loaded from two threads at once.

The loader registers a module in ``sys.modules`` before executing it, so a
sibling importing back into it finds it rather than re-running it. A second
*thread* arriving in that window used to be handed the same half-built module
and read an attribute the file had not defined yet — which presented as
``command_schedule.py has no 'ScheduleCommand'`` about a file that plainly has
one, whenever two sessions asked one command for its form at the same moment.

The failure is a race, so the test makes the window wide on purpose: the
fixture file sleeps *before* defining its class.
"""

import sys
import threading

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import sandbox  # noqa: F401,E402 - installs the bare ``guest`` alias
from guest.loader import load_entry, unload_box  # noqa: E402

SOURCE = """
import time
time.sleep(0.25)


class SlowCommand:
    name = "slow"
"""


def test_a_member_loaded_from_two_threads_is_never_seen_half_built(tmp_path):
    (tmp_path / "command_slow.py").write_text(SOURCE, encoding="utf-8")
    seen = []

    def load():
        try:
            seen.append(load_entry(str(tmp_path / "command_slow.py"),
                                   "SlowCommand", box_name="slow_box",
                                   bound=False))
        except BaseException as exc:      # noqa: BLE001 - the thing under test
            seen.append(exc)

    try:
        threads = [threading.Thread(target=load) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        unload_box("slow_box")

    assert [type(s).__name__ for s in seen] == ["SlowCommand"] * 4, seen
