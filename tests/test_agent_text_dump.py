"""The agent-text dump keeps pointing at real code.

``dev/dump_agent_text.py`` reads a curated list of kernel modules, because
"does this string reach a model" cannot be decided from the source and a
whole-tree scan buries the twenty sentences that matter under several hundred
log lines. Curation buys precision and costs drift: rename or move one of
those modules and the dump keeps working, keeps looking complete, and silently
stops showing a whole population.

That is the failure worth a test — not the formatting. A person editing the
agent's voice cannot tell the difference between "this text does not exist"
and "this file moved", and the dump exists precisely to be trusted on that
question.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dump():
    """The dev script, loaded by path — it is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "dump_agent_text", ROOT / "dev" / "dump_agent_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_curated_module_still_exists(dump):
    """The drift this design trades precision for."""
    missing = [relative for relative, _note in dump.KERNEL_SITES
               if not (ROOT / relative).is_file()]

    assert not missing, f"dump_agent_text points at moved/renamed files: {missing}"


def test_every_curated_module_still_yields_text(dump):
    """A file that exists but has gone quiet is the same failure one step on:
    the strings were moved out and the dump has nothing to say about them."""
    silent = [relative for relative, _note in dump.KERNEL_SITES
              if not dump.kernel_strings(relative)]

    assert not silent, f"no agent-facing strings found in: {silent}"


def test_log_lines_are_filtered_but_prose_survives(dump):
    """The one heuristic in the script, pinned against the module that most
    depends on it — two thirds of ``conversation_loop``'s long literals are log
    calls, and the tool-budget messages sit among them."""
    found = {text for _line, _fn, text in
             dump.kernel_strings("runtime/conversation_loop.py")}

    assert any("tool-call limit" in text for text in found)
    assert not any(text.startswith("subagent %s") for text in found)
    assert not any("%s" in text and "retry" in text.lower() for text in found)


def test_a_computed_leaf_does_not_lose_the_whole_schema(dump):
    """``literal_eval`` is all-or-nothing and one ``sorted(TYPES)`` used to
    flatten a seven-argument schema into an unreadable line."""
    source = (
        "parameters = {'type': 'object', 'properties': {"
        "'kind': {'type': 'string', 'enum': sorted(TYPES),"
        " 'description': 'Which kind.'},"
        "'note': {'type': 'string', 'description': 'Free text.'}},"
        " 'required': ['kind']}"
    )

    (_line, name, rendered), = dump.plugin_declarations(source)

    assert name == "parameters"
    assert "- kind: string (required)" in rendered
    assert "Which kind." in rendered
    assert "Free text." in rendered


def test_a_local_variable_is_not_a_declaration(dump):
    """Walking the whole tree collected every function-local ``name``, which
    is how ``name = (raw or '').strip()`` from a validator helper ended up
    listed as agent-facing text."""
    source = (
        "class T:\n"
        "    name = 'memory'\n"
        "    def run(self, sdk, raw):\n"
        "        name = (raw or '').strip()\n"
        "        description = 'not a declaration either'\n"
        "        return name + description\n"
    )

    found = dump.plugin_declarations(source)

    assert [(name, text) for _line, name, text in found] == [("name", "memory")]


def test_the_human_half_of_a_refusal_is_not_agent_text(dump):
    """``Decision`` carries ``reason`` for the ledger and the model, and
    ``say`` for the person reading the dialog. Only the first is ever shown to
    a model, so the second belongs in the user-facing dump."""
    found = {text for _line, _fn, text in dump.kernel_strings("sandbox/policy.py")}

    # A ``reason``: the model is told this when a script is refused.
    assert any("is not in a scripts/ directory" in text for text in found)
    # A ``say``: "Deleted rows are not recoverable." is dialog prose, and its
    # reason — "delete rows from {table}" — is what the model actually reads.
    assert not any("recoverable" in text for text in found)
    assert not any("allowed-hosts list" in text for text in found)


def test_a_dynamic_agent_prompt_is_named_rather_than_guessed(dump):
    """Its text depends on live state; inventing one would be worse than
    saying where it is."""
    source = "class T:\n    def agent_prompt(self, sdk):\n        return 'x'\n"

    (_line, name, rendered), = dump.plugin_declarations(source)

    assert name == "agent_prompt"
    assert rendered.startswith("<dynamic")


def test_nothing_boots_at_import(dump):
    """``live_view`` opens the real database and runs discovery, so importing
    this module — which this very test does — must do none of it."""
    tree = ast.parse((ROOT / "dev" / "dump_agent_text.py").read_text(encoding="utf-8"))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    guarded = [node for node in tree.body if isinstance(node, ast.If)]

    assert "live_view" not in called or guarded, "unguarded boot at module level"
    assert any(isinstance(node.test, ast.Compare)
               and getattr(node.test.left, "id", "") == "__name__"
               for node in guarded), "no __main__ guard"
