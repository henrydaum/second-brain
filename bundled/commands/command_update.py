"""Slash command plugin for `/update`."""

from guest.bases import BaseCommand

#: The web app, relative to the repo root. One line rather than a search,
#: which is the whole dividend of the UI living in this repo: the store's
#: ``/update_ui`` had to read a launch agent's plist to find a *second*
#: checkout, verify the path it got back was absolute, and pull it separately.
#: A pull of this repo is now a pull of the UI.
UI_DIR = "frame_ui"

#: How a platform deploys the UI, if it does. The value is a command to run in
#: ``frame_ui/``, and it is used **only when the script it names is present** —
#: absence is the ordinary case (a dev server needs no deployment) rather than
#: a misconfiguration, so it is not an error.
#:
#: Keyed on ``sys.platform`` as ``paths.get("platform")`` answers it, so
#: ``darwin`` rather than ``macos``.
DEPLOY = {
    "darwin": ("deploy/macos/manage.sh", ["sh", "deploy/macos/manage.sh", "update"]),
}

#: Per step. Deliberately not the 600 ``proc.run`` allows, because that is not
#: the binding limit: a command box is killed at ``watchdog.HARD_CEILING`` (600
#: seconds of **wall** clock, which is not discounted for time blocked on the
#: kernel), so two steps sharing one command have to fit inside it together. A
#: cold ``npm install`` is seconds and a release build is a minute or two; five
#: apiece leaves the ceiling reachable only by something that has genuinely
#: hung, which is the case a deadline is for.
UI_TIMEOUT = 300


class UpdateCommand(BaseCommand):
    """Slash-command handler for `/update`."""
    name = "update"
    description = "Pull latest changes from the Second Brain repo, and update the web UI"
    category = "System"
    require_approval = True
    approval_actor_id = "user"
    requests = ["paths.get", "proc.run", "fs.stat", "ui.progress"]

    def _git(self, sdk, root, *args):
        """Run a git subcommand in the repo root, returning stripped stdout."""
        result = sdk.proc.run(["git", *args], timeout=60, cwd=root)
        return (
            result["code"],
            (result.get("stdout") or "").strip(),
            (result.get("stderr") or "").strip(),
        )

    def run(self, sdk, args):
        """Execute `/update` for the active session."""
        try:
            root = sdk.paths.get("project")
            _, before, _ = self._git(sdk, root, "rev-parse", "HEAD")
            sdk.ui.progress("Pulling the repository...")
            code, out, err = self._git(sdk, root, "pull")
        except Exception as e:
            return f"Update failed: {e}"
        if code:
            return f"git pull failed (exit {code}):\n{err or out}"
        if not out or out.lower().startswith("already up to date"):
            return out or "Already up to date."
        _, after, _ = self._git(sdk, root, "rev-parse", "HEAD")
        if before == after:
            return out
        _, log, _ = self._git(
            sdk, root, "log", "--pretty=format:- %s", f"{before}..{after}"
        )
        summary = log or out
        # Only when the pull actually brought something: an npm install for a
        # tree that did not move is minutes spent to change nothing, and this
        # is the branch that already knows.
        ui = self._update_ui(sdk, root)
        return (f"Updated {before[:7]}..{after[:7]}:\n\n{summary}\n\n"
                f"{ui}\n\n/restart to take effect")

    # ──────────────────────────────────────────────────────────────────
    # The web UI.
    # ──────────────────────────────────────────────────────────────────

    def _update_ui(self, sdk, root):
        """Bring `frame_ui/` in step with the code that was just pulled.

        Two steps, and the first is the one that is always needed: the pull
        moved the app's source but not its `node_modules`, so a changed
        `package.json` leaves a dev server importing something that is not
        there. `npm install` is a no-op when nothing moved, which is what makes
        it safe to run unconditionally.

        The second is deployment, and it exists only where there is something
        to deploy — a dev server reloads itself, while the macOS install serves
        a *build* that has to be made and activated. `DEPLOY` says how per
        platform and the script's presence says whether; a platform with
        neither is not broken, so this reports rather than raises.

        Failures are reported, never raised. The kernel is already updated at
        this point and saying so is the more useful half of the answer — an
        exception here would replace the commit summary with a stack trace
        about npm.
        """
        ui_root = f"{root}/{UI_DIR}"
        if not sdk.fs.exists(f"{ui_root}/package.json"):
            return "Web UI: not present in this checkout, nothing to update."

        sdk.ui.progress("Updating web UI dependencies...")
        installed = self._run(sdk, ["npm", "install"], ui_root)
        if installed is not True:
            return f"Web UI: `npm install` failed.\n{_indent(installed)}"

        script, argv = DEPLOY.get(str(sdk.paths.get("platform")), (None, None))
        if script is None:
            return "Web UI: dependencies updated. Restart `npm run dev` to pick it up."
        if not sdk.fs.exists(f"{ui_root}/{script}"):
            # A platform we know how to deploy on, in a checkout that carries
            # no deployment. The dev server is the ordinary case; say what
            # happened rather than naming a missing file as a fault.
            return "Web UI: dependencies updated. Restart `npm run dev` to pick it up."

        sdk.ui.progress(f"Deploying the web UI ({' '.join(argv)})...")
        deployed = self._run(sdk, argv, ui_root)
        if deployed is not True:
            return f"Web UI: deployment failed.\n{_indent(deployed)}"
        return "Web UI: rebuilt and deployed. Reload the browser to use it."

    def _run(self, sdk, argv, cwd):
        """True, or the command's output for the caller to report."""
        try:
            # Through a shell, because ``npm`` on Windows is ``npm.cmd`` — a
            # batch file rather than an executable, so a bare exec fails
            # outright and reads as "npm is not installed" on a machine that
            # has it. ``"default"`` and not ``True``: the argument names *which
            # shell*, and the kernel renders the line, because ``cmd`` does not
            # parse what ``subprocess`` produces from a list.
            result = sdk.proc.run(argv, cwd=cwd, timeout=UI_TIMEOUT,
                                  shell="default")
        except sdk.Failed as e:
            return e.error
        if result["code"] == 0:
            return True
        # Each stream's tail on its own: a joined tail is whichever stream came
        # last, and a noisy stderr then pushes the compiler's stdout — the
        # actual failure — out of the report entirely.
        output = "\n".join((result.get(stream) or "").strip()[-2000:]
                           for stream in ("stdout", "stderr"))
        return f"exit {result['code']}\n{output.strip()}"


def _indent(text):
    """Keep arbitrary command output literal in markdown."""
    return "\n".join("    " + line for line in str(text).splitlines())
