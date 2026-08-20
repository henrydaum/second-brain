"""Approval options: the answers a dialog offers, and what they remember.

Every "allow once" used to be thrown away. These pin the two halves of fixing
that — a registry that stays open to answers nobody has thought of yet, and a
writer that turns one of them into an entry in a list the user keeps.

The refusals matter more than the grants. An option that writes to a list the
policy will then refuse to honour is worse than no option: it is a grant the
person believes they made.
"""

import os
import threading

import pytest

import paths
from runtime.context import set_kernel_parts
from sandbox import options
from sandbox.guest.requests import (FS_DELETE, FS_MOVE, FS_WRITE, NET_HTTP,
                                    PROC_RUN, Request)
from sandbox.policy import SAFE, Chain, classify

CHAIN = Chain(root="repl").push("tool_x")


@pytest.fixture(autouse=True)
def _blank_lists(monkeypatch):
    """Every test states its own starting point; none inherits the last.

    **And none of them touches the real ``config.json``.** ``remember`` writes
    through ``config_manager.save``, which resolves its own path from
    ``paths`` — so without this stub a test run appends its fixtures to the
    developer's actual egress allowlist. It did, once, which is why the stub
    records calls instead of merely swallowing them: "did we persist" is a
    thing these tests need to assert about.
    """
    from config import config_manager

    saved: list = []
    monkeypatch.setattr(config_manager, "save",
                        lambda config, path=None: saved.append(dict(config)))

    def blank():
        set_kernel_parts(config={"net_allowed_hosts": [],
                                 "fs_writable_dirs": [],
                                 "shell_allowed_prefixes": []})
    blank()
    yield saved
    blank()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Somebody's project folder, with the app moved out of the way.

    ``tmp_path`` is inside the repo under this project's pytest settings, so
    without moving ``ROOT_DIR`` the carve-out would suppress every folder
    option and the tests would pass for the wrong reason. Same fixture shape as
    ``tests/test_writable_dirs.py``; ``DATA_DIR`` is deliberately left alone
    (patching it poisons ``trees`` for the session).
    """
    app, work = tmp_path / "app", tmp_path / "work"
    app.mkdir()
    work.mkdir()
    monkeypatch.setattr(paths, "ROOT_DIR", app)
    return work


def _offered(request):
    """The option labels a dialog for ``request`` would show."""
    decision = classify(request, CHAIN)
    return [option.label for option in options.options_for(CHAIN, request, decision)]


def _grants(request):
    """Just the options that would remember something."""
    decision = classify(request, CHAIN)
    return [option for option in options.options_for(CHAIN, request, decision)
            if option.remember is not None]


# ── the registry ──────────────────────────────────────────────────────

def test_every_dialog_can_be_allowed_once_or_denied():
    """The two that are always there, in the order they are shown."""
    labels = _offered(Request(NET_HTTP, {"url": "https://example.invalid/"}))
    assert labels[0] == "Allow once"
    assert labels[-1] == "Deny"


def test_a_request_no_builder_recognises_offers_only_those_two():
    assert _offered(Request("plugin.install", {"stem": "tool_x"})) == [
        "Allow", "Deny"]


def test_a_lone_allow_drops_the_once_it_has_nothing_to_contrast_with():
    """"once" is only meaningful beside an "always".

    Every builder that offers a wider grant is conditional, so most dialogs
    show two buttons — and "Allow once" there distinguishes the option from
    nothing at all, while inviting the reader to wonder what the alternative
    was.
    """
    lone = Request("plugin.install", {"stem": "tool_x"})
    assert _offered(lone)[0] == "Allow"

    beside_an_always = Request(NET_HTTP, {"url": "https://example.invalid/"})
    assert _offered(beside_an_always)[0] == "Allow once"


def test_relabelling_a_lone_allow_leaves_the_answer_it_records_alone():
    """The label is for the reader; the value is for ``chosen`` and the ledger.

    A dialog built by an older version of this file is answered with "allow"
    and must still resolve, so the rename may never reach the value.
    """
    request = Request("plugin.install", {"stem": "tool_x"})
    offered = options.options_for(CHAIN, request, classify(request, CHAIN))

    assert [option.value for option in offered] == ["allow", "deny"]
    assert options.chosen(offered, "allow").allow is True


def test_a_raising_builder_costs_its_own_options_and_nothing_else():
    """Rule 2, and the same shape as a raising shell recognizer."""
    def boom(chain, request, decision):
        raise RuntimeError("builder bug")

    original = list(options.OPTION_BUILDERS)
    try:
        options.OPTION_BUILDERS.insert(0, boom)
        assert _offered(Request(NET_HTTP, {"url": "https://example.invalid/"})) == [
            "Allow once", "Always allow example.invalid", "Deny"]
    finally:
        options.OPTION_BUILDERS[:] = original


def test_an_option_with_no_value_is_never_built():
    """``_sane_enum`` would drop it anyway, misaligning every later label."""
    def blank(chain, request, decision):
        return [options.Option("", "Pressable but unanswerable")]

    original = list(options.OPTION_BUILDERS)
    try:
        options.OPTION_BUILDERS.insert(0, blank)
        assert "Pressable but unanswerable" not in _offered(
            Request(NET_HTTP, {"url": "https://example.invalid/"}))
    finally:
        options.OPTION_BUILDERS[:] = original


def test_chosen_maps_an_answer_back_and_refuses_an_unknown_one():
    """A restored session can answer a dialog an older build wrote."""
    offered = [options.ALLOW_ONCE, options.DENY]
    assert options.chosen(offered, "allow") is options.ALLOW_ONCE
    assert options.chosen(offered, "always:whatever") is None
    assert options.chosen(offered, None) is None


# ── hosts ─────────────────────────────────────────────────────────────

def test_the_host_option_names_the_host_and_not_the_domain():
    """``_host_allowed`` matches downward, so the domain grants strictly more."""
    grants = _grants(Request(NET_HTTP, {"url": "https://api.search.brave.com/x?q=1"}))
    assert [g.label for g in grants] == ["Always allow api.search.brave.com"]


def test_an_already_allowed_host_is_not_offered_again():
    set_kernel_parts(config={"net_allowed_hosts": ["brave.com"]})
    assert _grants(Request(NET_HTTP, {"url": "https://api.brave.com/x"})) == []


def test_a_url_with_no_host_offers_nothing_to_remember():
    assert _grants(Request(NET_HTTP, {"url": "not a url"})) == []


# ── folders ───────────────────────────────────────────────────────────

def test_the_folder_option_offers_the_targets_parent(project):
    grants = _grants(Request(FS_WRITE, {"path": str(project / "src" / "a.py")}))
    assert [g.label for g in grants] == [f"Always allow {project / 'src'}"]


def test_no_folder_is_offered_inside_the_apps_own_tree():
    """Rule 3. ``_freely_writable`` would refuse this grant, so offering it
    would be a button that lies."""
    assert _grants(Request(FS_WRITE,
                           {"path": str(paths.ROOT_DIR / "sandbox" / "policy.py")})) == []


def test_a_move_offers_both_ends_because_classify_needs_both(project):
    """Offering only the destination hands back a grant that changes nothing."""
    grants = _grants(Request(FS_MOVE, {"src": str(project / "a" / "x.txt"),
                                       "dst": str(project / "b" / "x.txt")}))
    assert [g.label for g in grants] == [f"Always allow {project / 'a'}",
                                         f"Always allow {project / 'b'}"]


def test_a_move_within_one_folder_offers_it_once(project):
    grants = _grants(Request(FS_MOVE, {"src": str(project / "x.txt"),
                                       "dst": str(project / "y.txt")}))
    assert len(grants) == 1


def test_an_already_writable_folder_is_not_offered(project):
    set_kernel_parts(config={"fs_writable_dirs": [str(project)]})
    assert _grants(Request(FS_DELETE, {"path": str(project / "a.txt")})) == []


def test_a_missing_path_offers_nothing():
    assert _grants(Request(FS_WRITE, {"path": ""})) == []


# ── commands ──────────────────────────────────────────────────────────

def test_the_command_option_offers_the_program_and_subcommand():
    grants = _grants(Request(PROC_RUN, {"argv": ["git", "push", "--force"]}))
    assert [g.label for g in grants] == ["Always allow: git push"]


def test_a_program_whose_job_is_running_other_code_is_never_offered():
    """The deny-set, and why it is about the *button* and not the matcher.

    A granted verb runs with whatever follows it — the bargain the setting
    states. For these the bargain is unbounded by construction: their whole
    job is to execute something named elsewhere, so "always allow python" is
    the shell again under another name. ``command_prefix`` still reduces
    ``python train.py`` honestly (see ``tests/test_shell_recognizer.py``);
    there is simply no one-click way to grant it.
    """
    for argv in (["python", "train.py"], ["npm", "install", "x"],
                 ["bash", "-c", "date"], ["sudo", "apt", "update"],
                 ["make", "build"]):
        assert _grants(Request(PROC_RUN, argv and {"argv": argv})) == [], argv


def test_config_remains_the_way_to_grant_one_anyway():
    """The deny-set gates the dialog, not the allowlist.

    Somebody who genuinely wants ``python`` unattended can put it in
    ``shell_allowed_prefixes`` with ``/config`` — a considered act, rather than
    one click in the middle of a turn.
    """
    from sandbox.policy import SAFE
    set_kernel_parts(config={"shell_allowed_prefixes": ["python"]})
    assert classify(Request(PROC_RUN, {"argv": ["python", "train.py"]}),
                    CHAIN).level == SAFE


def test_no_command_option_when_there_is_no_unit_to_describe():
    """Everything the read-only recognizer refuses to look at."""
    for args in ({"argv": ["git", "push", "|", "tee", "log"]},
                 {"argv": "git push > ~/out.txt", "shell": "cmd"},
                 {"argv": ["/usr/bin/git", "push"]},
                 {"argv": []}):
        assert _grants(Request(PROC_RUN, args)) == [], args


def test_a_posix_shell_decomposes_where_cmd_cannot():
    """The same Request answers differently by platform, on purpose.

    ``shell="default"`` is ``/bin/sh`` on POSIX and ``cmd.exe`` on Windows, and
    ``_posix_shell`` asks the platform rather than trusting the name — so this
    line has two units on one and none the lexer will describe on the other.
    Where it decomposes, the grant has to cover *both*, which is the module's
    own worked example: ``git push && rm -rf /`` needs ``rm`` granted too.

    This lived in the list above as a case that offers nothing, which was true
    only on the machine it was written on.
    """
    grants = _grants(Request(PROC_RUN, {"argv": "git push && rm -rf /",
                                        "shell": "default"}))
    if os.name == "nt":
        assert grants == []
    else:
        assert [option.value for option in grants] == ["always:git push, rm"]


def test_a_command_through_a_shell_is_still_offered_when_the_line_is_inert():
    """``tool_run_command`` always names a shell, so without this the option
    never appeared for any command at all."""
    grants = _grants(Request(PROC_RUN, {"argv": "git pull", "shell": "default"}))
    assert [g.label for g in grants] == ["Always allow: git pull"]


# ── the rest of this turn ─────────────────────────────────────────────
#
# The one builder whose unit is *time*, and the only one that can offer a
# grant whose scope the person cannot check for themselves: nothing appears in
# ``/config`` afterwards, because it is gone before they could look. So what
# these pin is the window — offered inside a turn, and nowhere else.


class _FakeRuntime:
    """Just the three questions ``_rest_of_this_turn`` asks a runtime."""

    def __init__(self, busy=True, mode="ask"):
        self.busy, self.mode, self.set = busy, mode, []

    def is_turn_in_flight(self, key):
        return self.busy

    def security_mode(self, key):
        return self.mode

    def set_security_mode(self, key, mode, *, scope="conversation"):
        self.set.append((key, mode, scope))
        return mode


@pytest.fixture
def kernel_runtime(monkeypatch):
    """Install a runtime for the duration of one test, and only that long.

    ``set_kernel_parts`` ignores ``None``, so it cannot un-install one — a
    fake left behind would answer every later test's policy question.
    """
    from runtime.context import _KERNEL_PARTS

    def install(runtime):
        monkeypatch.setitem(_KERNEL_PARTS, "runtime", runtime)
        return runtime
    return install


_TURN_LABEL = "Allow, and stop asking for the rest of this turn"
_ASKING = Request(NET_HTTP, {"url": "https://example.invalid/"})


def _turn_option(chain):
    built = options.options_for(chain, _ASKING, classify(_ASKING, chain))
    return [option for option in built if option.label == _TURN_LABEL]


def test_the_turn_option_is_offered_during_an_agent_turn(kernel_runtime):
    kernel_runtime(_FakeRuntime(busy=True))
    assert _turn_option(CHAIN)


def test_no_turn_option_when_no_agent_turn_is_running(kernel_runtime):
    """The case a session key alone cannot tell apart, and the reason this
    builder needs two questions rather than one.

    A frontend acting as one of its sessions (``sdk.frontend.act`` — a button
    in a web client) roots at that session, with a person right there watching
    it, and no turn anywhere. Offered here the grant is dropped by the *next*
    turn's end: whenever they happen to send a message and the agent happens
    to finish replying. A button saying "the rest of this turn" would be
    naming a scope nobody could have predicted from it.
    """
    kernel_runtime(_FakeRuntime(busy=False))
    assert _turn_option(Chain(root="http:7:7:0").push("frontend:http")) == []


def test_a_typed_command_never_reaches_the_turn_option(kernel_runtime):
    """The half that already worked: ``user:command`` names no session."""
    kernel_runtime(_FakeRuntime(busy=True))
    assert _turn_option(Chain(root="user:command").push("packages")) == []


def test_no_turn_option_when_the_turn_is_already_in_yolo(kernel_runtime):
    """Rule 3: never offer what changes nothing."""
    kernel_runtime(_FakeRuntime(busy=True, mode="yolo"))
    assert _turn_option(CHAIN) == []


def test_an_unwired_kernel_offers_no_turn_option():
    assert _turn_option(CHAIN) == []


def test_the_turn_option_writes_a_turn_scoped_yolo(kernel_runtime):
    """It writes to the *session*, never to config — which is what makes an
    expiring grant expressible at all."""
    runtime = kernel_runtime(_FakeRuntime(busy=True))
    assert _turn_option(CHAIN)[0].remember() is True
    assert runtime.set == [(CHAIN.root, "yolo", "turn")]


# ── remembering ───────────────────────────────────────────────────────

def test_remembering_makes_the_same_request_safe(project):
    """The property the whole feature exists for, one per list."""
    net = Request(NET_HTTP, {"url": "https://api.search.brave.com/x"})
    _grants(net)[0].remember()
    assert classify(net, CHAIN).level == SAFE

    write = Request(FS_WRITE, {"path": str(project / "src" / "a.py")})
    _grants(write)[0].remember()
    assert classify(write, CHAIN).level == SAFE

    run = Request(PROC_RUN, {"argv": ["git", "push", "--force"]})
    _grants(run)[0].remember()
    assert classify(run, CHAIN).level == SAFE


def test_a_no_op_grant_persists_nothing_and_says_so(_blank_lists):
    """No save means no ledger row and no "settings changed" notice."""
    saved = _blank_lists
    assert options.remember("net_allowed_hosts", "example.com") is True
    assert options.remember("net_allowed_hosts", "example.com") is False
    assert options.remember("net_allowed_hosts", "api.example.com") is False
    assert len(saved) == 1, "a no-op grant went to disk"


def test_a_failed_save_does_not_leave_a_live_unpersisted_grant(_blank_lists):
    """It would come back as a surprise at the next restart, widened."""
    from config import config_manager
    from runtime.context import kernel_config

    def boom(config, path=None):
        raise OSError("disk full")

    config_manager.save = boom
    assert options.remember("net_allowed_hosts", "example.com") is False
    assert kernel_config()["net_allowed_hosts"] == []


def test_granting_a_parent_folder_tidies_away_its_children(tmp_path):
    """Subsumption is what keeps the list readable after a week of clicking."""
    parent, child = tmp_path / "proj", tmp_path / "proj" / "src"
    child.mkdir(parents=True)
    assert options.remember("fs_writable_dirs", str(child)) is True
    assert options.remember("fs_writable_dirs", str(parent)) is True
    from runtime.context import kernel_config
    assert kernel_config()["fs_writable_dirs"] == [str(parent.resolve())]


def test_granting_a_domain_tidies_away_its_subdomains():
    options.remember("net_allowed_hosts", "api.example.com")
    options.remember("net_allowed_hosts", "example.com")
    from runtime.context import kernel_config
    assert kernel_config()["net_allowed_hosts"] == ["example.com"]


def test_a_hand_typed_comma_string_is_not_discarded_by_the_first_grant():
    """Both lists accept the string form, so a grant must not eat it."""
    set_kernel_parts(config={"net_allowed_hosts": "a.example, b.example"})
    options.remember("net_allowed_hosts", "c.example")
    from runtime.context import kernel_config
    assert kernel_config()["net_allowed_hosts"] == ["a.example", "b.example",
                                                    "c.example"]


def test_an_unwired_kernel_writes_nothing():
    """``kernel_config()`` answers ``{}``; ``save({})`` would write DEFAULTS
    over a real user's file."""
    set_kernel_parts(config={})
    from runtime.context import _KERNEL_PARTS
    _KERNEL_PARTS["config"] = {}
    assert options.remember("net_allowed_hosts", "example.com") is False


def test_every_option_targets_a_real_kernel_setting():
    """A typo'd key would be saved and then read by nobody, forever."""
    from config.config_manager import DEFAULTS

    assert set(options.MERGERS) <= set(DEFAULTS)


def test_concurrent_grants_do_not_lose_each_other():
    """The classic lost update: two hosts approved in the same second."""
    hosts = [f"h{i}.example.com" for i in range(24)]
    barrier = threading.Barrier(len(hosts))

    def grant(host):
        barrier.wait()
        options.remember("net_allowed_hosts", host)

    threads = [threading.Thread(target=grant, args=(host,)) for host in hosts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    from runtime.context import kernel_config
    assert sorted(kernel_config()["net_allowed_hosts"]) == sorted(hosts)
