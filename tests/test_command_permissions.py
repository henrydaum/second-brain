"""`/permissions`: the grants, and the two things the grants do not say.

The whole approval design rests on a standing grant being enumerable and
withdrawable — an answer nobody can find is an answer nobody can take back.
Before this the three settings holding every grant were three keys among
thirty in ``/config``, reachable only by knowing their names.

Driven through the module's own helpers and a stub SDK rather than a live
form, the way ``tests/test_command_menus.py`` does: the decisions are in the
helpers, and the assertions stay readable.
"""

import importlib.util
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias

_COMMANDS = Path(__file__).resolve().parents[1] / "bundled" / "commands"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_perm_command", _COMMANDS / "command_permissions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Config:
    """The three settings, readable and writable."""

    def __init__(self, values):
        self.values = dict(values)
        self.writes = []

    def read(self, key, **kwargs):
        return self.values.get(key)

    def write(self, key, value, **kwargs):
        self.values[key] = value
        self.writes.append((key, value))


class _Markdown:
    """Enough of ``sdk.md`` to render the listing."""

    @staticmethod
    def table(headers, rows, **kwargs):
        lines = ["| " + " | ".join(str(h) for h in headers) + " |",
                 "|" + "|".join("---" for _ in headers) + "|"]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        return "\n" + "\n".join(lines)


class _SDK:
    def __init__(self, **values):
        self.config = _Config(values)
        self.md = _Markdown()


@pytest.fixture
def module():
    return _load()


@pytest.fixture
def sdk():
    return _SDK(net_allowed_hosts=["api.search.brave.com", "example.com"],
                fs_writable_dirs=[r"Z:\My Code\Proj"],
                shell_allowed_prefixes=["git pull", "pytest"])


def _run(module, sdk, **args):
    return module.PermissionsCommand().run(sdk, args)


def _form(module, sdk, **args):
    """The form's steps. Guest ``FormStep`` is a dict subclass, so these are
    read by key rather than by attribute."""
    return module.PermissionsCommand().form(sdk, args)


# ── the listing ───────────────────────────────────────────────────────

def test_the_overview_shows_every_grant_from_every_list(module, sdk):
    out = _run(module, sdk)
    for entry in ("api.search.brave.com", "example.com", r"Z:\My Code\Proj",
                  "git pull", "pytest"):
        assert entry in out, entry


def test_an_empty_list_is_omitted_rather_than_shown_empty(module):
    out = _run(module, _SDK(net_allowed_hosts=["example.com"]))
    assert "Network" in out
    assert "Writable folders" not in out
    assert "Commands" not in out


def test_the_listing_says_what_it_is_not_showing(module, sdk):
    """The two footnotes answer "what can it reach", which the tables do not.

    Without them the view is wrong in both directions: the agent is freer than
    the lists suggest inside its own tree, and more constrained near the app's
    own files. Both are pinned because a listing that quietly stops saying
    this is a listing people will trust for something it does not answer.
    """
    for out in (_run(module, sdk), _run(module, _SDK())):
        assert "workspace" in out
        assert "Never grantable" in out


def test_no_grants_at_all_is_a_sentence_not_an_empty_page(module):
    out = _run(module, _SDK())
    assert "No standing permissions" in out
    assert "asked about individually" in out


def test_a_host_entry_says_that_a_domain_covers_its_subdomains(module):
    """The only part of an egress grant a person could be surprised by."""
    out = _run(module, _SDK(net_allowed_hosts=["example.com"]))
    assert "subdomain" in out


def test_a_folder_entry_says_deletes_are_included(module):
    """It is the grant's sharpest edge and the easiest thing to forget."""
    out = _run(module, _SDK(fs_writable_dirs=["/srv/proj"]))
    assert "delete" in out.lower()


# ── revoking ──────────────────────────────────────────────────────────

def test_revoking_one_entry_leaves_the_others(module, sdk):
    out = _run(module, sdk, category="Network", entry="example.com")

    assert sdk.config.writes == [
        ("net_allowed_hosts", ["api.search.brave.com"])]
    assert "Revoked" in out


def test_revoking_all_clears_only_that_list(module, sdk):
    _run(module, sdk, category="Commands", entry=module._REVOKE_ALL)

    assert sdk.config.values["shell_allowed_prefixes"] == []
    assert sdk.config.values["net_allowed_hosts"] == ["api.search.brave.com",
                                                     "example.com"]


def test_an_entry_that_is_not_there_writes_nothing(module, sdk):
    """A stale form answer must not silently rewrite the list."""
    out = _run(module, sdk, category="Network", entry="gone.example")

    assert sdk.config.writes == []
    assert "not in" in out


def test_an_unknown_category_writes_nothing(module, sdk):
    out = _run(module, sdk, category="Something else", entry="x")
    assert sdk.config.writes == []
    assert "Unknown" in out


# ── the form ──────────────────────────────────────────────────────────

def test_one_populated_list_skips_the_category_step(module):
    """Not a choice worth making, and skipping it also skips the re-ask."""
    steps = _form(module, _SDK(shell_allowed_prefixes=["git pull"]))

    assert [step["name"] for step in steps] == ["entry"]
    assert "git pull" in steps[0]["enum"]


def test_several_populated_lists_ask_which_one_and_count_them(module, sdk):
    steps = _form(module, sdk)

    assert steps[0]["name"] == "category"
    assert "Network (2)" in steps[0]["enum_labels"]


def test_the_entry_step_offers_a_revoke_all(module, sdk):
    steps = _form(module, sdk, category="Network")

    entry = next(step for step in steps if step["name"] == "entry")
    assert entry["enum"][-1] == module._REVOKE_ALL
    assert "api.search.brave.com" in entry["enum"]


def test_nothing_granted_means_no_form_at_all(module):
    assert _form(module, _SDK()) == []


# ── reading the setting ───────────────────────────────────────────────

def test_a_hand_typed_comma_string_lists_as_separate_grants(module):
    """Both settings document that form, and a hand-edited file is read
    before it is ever normalized again. One long row would be a lie."""
    out = _run(module, _SDK(net_allowed_hosts="a.example, b.example"))
    assert "a.example" in out and "b.example" in out


def test_blank_entries_are_not_offered_as_revocable(module):
    steps = _form(module, _SDK(shell_allowed_prefixes=["git pull", "", "  "]))
    entry = next(step for step in steps if step["name"] == "entry")
    assert entry["enum"] == ["git pull", module._REVOKE_ALL]


# ── the declaration ───────────────────────────────────────────────────

def test_the_command_declares_its_write_up_front(module):
    """``config.write`` is consequential, so the state machine asks before the
    body runs rather than leaving it to the execution-time approver.

    ``tests/test_command_approval_declarations.py`` derives that requirement
    from ``policy.ALWAYS_UNSAFE``; this pins the literal, since declarations
    are read by AST and a computed tuple reads as nothing at all.
    """
    command = module.PermissionsCommand
    assert command.approval_actions == ("Revoke",)
    assert "config.write" in command.requests


def test_this_view_lists_every_setting_a_grant_can_be_written_to(module):
    """The drift that would fail in silence.

    ``options.MERGERS`` is the complete set of settings an approval dialog can
    write a grant into; ``SETTINGS`` is what this view can show and revoke. A
    fourth grant type added to the first and forgotten in the second is a
    permission nobody can find and therefore nobody can withdraw — with no
    error anywhere, which is the whole failure mode this command exists to
    prevent. Derived rather than restated, so the assertion cannot rot.
    """
    from sandbox.options import MERGERS

    assert set(module.SETTINGS.values()) == set(MERGERS)


def test_every_listed_setting_is_a_real_kernel_setting(module):
    """A typo'd key would be revocable here and read by nobody."""
    from config.config_manager import DEFAULTS

    assert set(module.SETTINGS.values()) <= set(DEFAULTS)
