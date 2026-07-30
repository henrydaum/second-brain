"""Fixtures shared across the suite.

The doubles themselves live in ``tests/support.py`` so a test can import them
at module scope; this file only wraps them as fixtures for the common cases.

The repo-root ``conftest.py`` is a different thing and stays where it is: it
redirects pytest's temp root and resets the LLM registry between tests, both of
which have to apply to the whole run rather than to ``tests/`` alone.
"""

import pytest

from tests.support import FakeLLM, make_runtime


@pytest.fixture
def fake_llm():
    """Build a :class:`FakeLLM` from a list of queued responses."""
    return FakeLLM


@pytest.fixture
def conv_runtime(tmp_path):
    """Factory for a runtime on a throwaway database.

    A factory rather than a plain fixture because tests want to queue the
    model's answers before the turn runs, and several open two runtimes to
    check that one user cannot see the other's rows.

        rt, session, llm = conv_runtime([response("hi")])

    Deliberately not named ``runtime``: several files already define a local
    fixture by that name meaning something narrower, and a conftest fixture
    that silently shadows one of them would be answered from the wrong rig.
    """
    def build(responses=None, **kwargs):
        return make_runtime(tmp_path, responses, **kwargs)
    return build


@pytest.fixture
def sandbox_box():
    """A sandbox that refuses everything unsafe, shut down afterwards."""
    from sandbox import Sandbox

    made = Sandbox()
    yield made
    made.shutdown()
