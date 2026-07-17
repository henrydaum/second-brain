"""Slash command plugin for `/update`."""

import subprocess

from plugins.BaseCommand import BaseCommand


class UpdateCommand(BaseCommand):
    """Slash-command handler for `/update`."""
    name = "update"
    description = "Pull latest changes from the Second Brain repo"
    category = "Config & System"
    require_approval = True
    approval_actor_id = "user"

    def _git(self, context, *args):
        """Run a git subcommand in the repo root, returning stripped stdout."""
        result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=60, cwd=context.root_dir)
        return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()

    def run(self, args, context):
        """Execute `/update` for the active session."""
        try:
            _, before, _ = self._git(context, "rev-parse", "HEAD")
            code, out, err = self._git(context, "pull")
        except Exception as e:
            return f"Update failed: {e}"
        if code:
            return f"git pull failed (exit {code}):\n{err or out}"
        if not out or out.lower().startswith("already up to date"):
            return out or "Already up to date."
        _, after, _ = self._git(context, "rev-parse", "HEAD")
        if before == after:
            return out
        _, log, _ = self._git(context, "log", "--pretty=format:- %s", f"{before}..{after}")
        summary = log or out
        return f"Updated {before[:7]}..{after[:7]}:\n\n{summary}\n\n/restart to take effect"
