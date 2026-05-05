import subprocess

from plugins.BaseCommand import BaseCommand


class UpdateCommand(BaseCommand):
    name = "update"
    description = "Pull latest changes from the Second Brain repo"
    category = "Config & System"
    require_approval = True
    approval_actor_id = "user"

    def run(self, args, context):
        try:
            result = subprocess.run(["git", "pull"], capture_output=True, text=True, timeout=60, cwd=context.root_dir)
        except Exception as e:
            return f"Update failed: {e}"
        out, err = (result.stdout or "").strip(), (result.stderr or "").strip()
        if result.returncode:
            return f"git pull failed (exit {result.returncode}):\n{err or out}"
        if not out or out.lower().startswith("already up to date"):
            return out or "Already up to date."
        return f"{out}\n\n/restart to take effect"
