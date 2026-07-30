"""Shared test doubles.

These existed five times over, copied between files as each new test needed a
fake model. The copies drifted — some recorded calls as bare message lists,
some as dicts with kwargs; some raised when the queued responses ran out and
some kept answering — so a change to ``ConversationRuntime``'s signature was a
fifteen-file edit and a change to the loop's call shape broke whichever copies
happened to care.

Anything here is deliberately a *superset* of what the copies did: declaring an
attribute no test reads costs nothing, while a fake missing one produces a
failure that looks like a kernel bug. Fixtures wrapping these live in
``conftest.py``; import the classes directly when a test needs one at module
scope.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def response(content="", tool_calls=None, **extra):
    """One model response, shaped like what a backend returns.

    ``extra`` covers the occasional test that needs ``is_error`` or a token
    count, without every caller restating the full shape.
    """
    fields = dict(
        content=content,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        is_error=False,
        prompt_tokens=0,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


class FakeLLM:
    """Answers from a queued list, and records what it was asked.

    Runs out gracefully: once the queue empties it keeps answering ``"done."``.
    A turn that loops one more time than the test expected should fail on the
    transcript it produced, not on an ``IndexError`` from the fake — the second
    tells you nothing about what went wrong.
    """

    # 0 disables proactive compaction, which is what nearly every test wants;
    # the compaction tests set it explicitly.
    context_size = 0
    model_name = "fake"
    loaded = True
    # Attachment routing is the kernel's job and it asks the model what it can
    # read. A fake declaring nothing gets the text fallback for everything —
    # correct, but it would make a vision test assert nothing.
    capabilities = {"image": True, "audio": True, "video": True}

    def __init__(self, responses=None):
        """Queue zero or more responses."""
        self._responses = list(responses or [])
        # Bare message lists — the shape most assertions want.
        self.calls = []
        # The same calls with everything else that came along, for the tests
        # that assert on tools or provider kwargs.
        self.records = []
        self.attachments = []

    def chat_with_tools(self, messages, tools=None, attachments=None, **kwargs):
        """Record the call and answer with the next queued response."""
        self.calls.append(list(messages))
        self.records.append({"messages": list(messages), "tools": tools,
                             "attachments": attachments, "kwargs": kwargs})
        self.attachments.append(attachments)
        if self._responses:
            return self._responses.pop(0)
        return response(content="done.")


class ToolChoiceLLM(FakeLLM):
    """A model that admits to supporting ``tool_choice``."""

    supports_tool_choice = True


class FakeRegistry:
    """A tool registry that only knows how to hand out schemas."""

    tools = {}  # empty -> no per-tool budget enforcement

    def __init__(self, schemas=None, max_tool_calls=5):
        """Hold the schemas the agent should see."""
        self._schemas = list(schemas or [])
        self.max_tool_calls = max_tool_calls

    def get_all_schemas(self):
        """Every schema the agent may call."""
        return self._schemas


def agent_state(tools=None, cache=None):
    """A ConversationState with turn priority already on the agent."""
    from state_machine.conversation import ConversationState, Participant
    from state_machine.conversation_phases import BASE_PHASE

    base = {"session_key": "chat",
            "agent_scoped_tool_names": list((tools or {}).keys())}
    base.update(cache or {})
    return ConversationState(
        [Participant("user", "user"),
         Participant("agent", "agent", tools=tools or {})],
        "agent", BASE_PHASE, base)


def plain_runtime(db, **kwargs):
    """A ConversationRuntime on an existing database, with no model.

    The single most repeated line in the suite — twenty-two copies of
    ``ConversationRuntime(db=db, services={}, config={})`` across seven files,
    which is why adding one required argument used to be a seven-file edit.
    Tests that need a model want :func:`make_runtime` instead.
    """
    from runtime.conversation_runtime import ConversationRuntime

    kwargs.setdefault("services", {})
    kwargs.setdefault("config", {})
    return ConversationRuntime(db=db, **kwargs)


def make_runtime(tmp_path, responses=None, *, name="test.db", services=None,
                 config=None, session_key="s", title="x", **kwargs):
    """A ConversationRuntime on a fresh database with one open conversation.

    Returns ``(runtime, session, llm)``. The ``llm`` is the :class:`FakeLLM`
    that was installed unless ``services`` overrode it, so a test can queue
    responses and then read back what the loop actually sent.
    """
    from pipeline.database import Database
    from runtime.conversation_runtime import ConversationRuntime

    db = Database(str(tmp_path / name))
    cid = db.create_conversation(title)
    llm = FakeLLM(responses)
    if services is None:
        services = {"llm": llm}
    runtime = ConversationRuntime(db=db, services=services,
                                  config=config or {}, **kwargs)
    session = runtime.load_conversation(session_key, cid)
    return runtime, session, llm


# ── Reaching the store branch ─────────────────────────────────────────
#
# Some kernel invariants are about *store* files: what the validator says
# about them, and what declarations the bridge reads off them. Those files are
# not in this tree, so they are materialized from the store branch.
#
# A worktree is preferred over the committed ref when the clone has one.
# Mid-migration the interesting version of a file is the one being edited, and
# a test that silently checks the last commit instead is a test that passes
# while the work is broken.

def store_worktree():
    """A checkout of the store branch, if this clone has one."""
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "list", "--porcelain"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        encoding="utf-8", check=False)
    root = None
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            root = Path(line.split(" ", 1)[1])
        elif line.strip() in {"branch refs/heads/store", "branch store"}:
            return root
    return None


def store_source(relative: str):
    """The store's copy of one file, from a worktree or a ref, or None."""
    worktree = store_worktree()
    if worktree is not None and (worktree / relative).is_file():
        return (worktree / relative).read_text(encoding="utf-8")
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{ref}:{relative}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None
