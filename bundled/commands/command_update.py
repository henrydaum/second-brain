"""Slash command plugin for `/update`."""

from guest.bases import BaseCommand


class UpdateCommand(BaseCommand):
    """Slash-command handler for `/update`."""
    name = "update"
    description = "Pull latest changes from the Second Brain repo"
    category = "Config & System"
    require_approval = True
    approval_actor_id = "user"
    requests = ["paths.get", "proc.run"]

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
        return f"Updated {before[:7]}..{after[:7]}:\n\n{summary}\n\n/restart to take effect"
