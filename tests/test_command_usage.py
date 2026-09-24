"""`/usage`: everything on the first screen, deeper views behind buttons.

Driven through the module and a stub SDK, the way ``test_command_permissions``
does, except that ``sdk.usage.read`` answers from the real ``usage.read``
handler over a real database — so what is asserted is what a person would see.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from pipeline.database import Database
from sandbox.guest.requests import USAGE_READ
from sandbox.guest.sdk import _Markdown
from tests.support import call_handler

_COMMANDS = Path(__file__).resolve().parents[1] / "bundled" / "commands"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_usage_command", _COMMANDS / "command_usage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SDK:
    Failed = Exception

    def __init__(self, db, conversation_id=None, retention=0):
        ctx = SimpleNamespace(db=db, user_id=1, session_key="chat",
                              runtime=SimpleNamespace(sessions={
                                  "chat": SimpleNamespace(
                                      conversation_id=conversation_id)}))
        self.md = _Markdown()
        self.config = SimpleNamespace(
            read=lambda key: retention if key == "data_retention_days" else None)

        def read(conversation_id=None, **kwargs):
            result = call_handler(USAGE_READ, ctx, {
                "conversation_id": conversation_id, **kwargs})
            assert result.ok, result.error
            return result.data

        self.usage = SimpleNamespace(read=read)


def _db(tmp_path):
    db = Database(str(tmp_path / "usage.db"))
    cid = db.create_conversation("Planning the trip")
    db.record_llm_usage(origin="agent", ok=True, user_id=1, conversation_id=cid,
                        model="big", input_tokens=10_000,
                        cache_read_tokens=7_500, output_tokens=400)
    db.record_llm_usage(origin="compactor", ok=True, user_id=1,
                        conversation_id=cid, model="small",
                        input_tokens=2_000, output_tokens=100)
    return db, cid


def test_the_first_step_is_the_overview(tmp_path):
    db, cid = _db(tmp_path)
    module = _load()

    [step] = module.UsageCommand().form(_SDK(db, cid), {})
    prompt = step["prompt"]

    assert "This conversation" in prompt
    assert "7,500 (62.5% of input)" in prompt      # cache hit, as a share
    assert "12,500" in prompt                       # total = input + output
    # Context now is the agent's last prompt, not the compactor's.
    assert "10,000 (last call)" in prompt
    assert "| small |" in prompt                    # split by model
    for period in ("Today", "Last 7 days", "All time"):
        assert period in prompt
    assert step["enum"] == list(module.VIEWS)


def test_no_conversation_still_shows_the_periods(tmp_path):
    db, _ = _db(tmp_path)
    module = _load()

    text = module.UsageCommand().run(_SDK(db, None), {})

    assert "This conversation" not in text
    assert "All time" in text


def test_nothing_recorded_says_so(tmp_path):
    module = _load()

    text = module.UsageCommand().run(
        _SDK(Database(str(tmp_path / "empty.db"))), {})

    assert "No model calls recorded yet." in text


def test_deeper_views(tmp_path):
    db, cid = _db(tmp_path)
    module = _load()
    command, sdk = module.UsageCommand(), _SDK(db, None, retention=30)

    assert "Planning the trip" in command.run(sdk, {"view": "conversations"})
    assert "| compactor |" in command.run(sdk, {"view": "sources"})
    assert "| big |" in command.run(sdk, {"view": "models"})
    days = command.run(sdk, {"view": "days"})
    assert "**Total**" in days and "kept for 30 days" in days
    assert f"Conversation {cid}" in command.run(sdk, {"view": str(cid)})
    assert "Unknown view" in command.run(sdk, {"view": "nonsense"})


def test_an_unreported_count_is_a_dash_not_a_zero(tmp_path):
    module = _load()
    db = Database(str(tmp_path / "quiet.db"))
    db.record_llm_usage(origin="agent", ok=True, user_id=1)

    text = module.UsageCommand().run(_SDK(db), {"view": "models"})

    assert "| — |" in text
    assert "did not report token counts" in text
