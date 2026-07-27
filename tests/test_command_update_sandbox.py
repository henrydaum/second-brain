"""SDK and parity coverage for the sandboxed ``/update`` command."""

import subprocess
from types import SimpleNamespace

from sandbox.facade import Sandbox
from sandbox.handlers.kernel import _path_get
from sandbox.policy import Chain, SAFE, classify
from sandbox.guest.requests import PATH_GET, Request


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def test_paths_get_exposes_only_named_application_locations(tmp_path):
    ctx = SimpleNamespace(root_dir=tmp_path)

    project = _path_get(ctx, {"name": "project"})
    assert project.ok
    assert project.data == str(tmp_path)

    unknown = _path_get(ctx, {"name": "arbitrary"})
    assert not unknown.ok
    assert "expected one of" in unknown.error

    request = Request(PATH_GET, {"name": "project"})
    assert request.read_only
    assert classify(request, Chain().push("update")).level == SAFE


def test_update_command_runs_in_an_up_to_date_clone(tmp_path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"

    _git("init", "--bare", str(remote), cwd=tmp_path)
    _git("clone", str(remote), str(seed), cwd=tmp_path)
    _git("config", "user.email", "tests@example.invalid", cwd=seed)
    _git("config", "user.name", "Tests", cwd=seed)
    (seed / "README.md").write_text("fixture\n", encoding="utf-8")
    _git("add", "README.md", cwd=seed)
    _git("commit", "-m", "fixture", cwd=seed)
    _git("push", "origin", "HEAD", cwd=seed)
    _git("clone", str(remote), str(checkout), cwd=tmp_path)

    context = SimpleNamespace(root_dir=checkout)
    sandbox = Sandbox(context=context, approve=lambda *_: True)
    try:
        result = sandbox.run(
            "plugins/commands/command_update.py",
            "UpdateCommand",
            kwargs={"args": {}},
        )
    finally:
        sandbox.shutdown()

    assert result.ok, result.error
    assert result.data == "Already up to date."
