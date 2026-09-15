"""Update an installed macOS Second Brain UI checkout and deployment."""

from guest.bases import BaseCommand

dependencies_files = []
dependencies_pip = []


class UpdateUICommand(BaseCommand):
    """Optional UI updater, separate from the kernel's /update command."""

    name = "update_ui"
    description = "Pull, verify, and deploy the macOS Second Brain UI"
    category = "System"
    require_approval = True
    approval_actor_id = "user"
    requests = ["paths.get", "proc.run", "ui.progress"]

    def _run(self, sdk, argv, stage, cwd=None, timeout=60):
        result = sdk.proc.run(argv, cwd=cwd, timeout=timeout)
        output = "\n".join(filter(None, [result.get("stdout"), result.get("stderr")]))
        if result["code"] != 0:
            raise RuntimeError(
                f"{stage} failed (exit {result['code']}).\n{output[-12000:]}"
            )
        return (result.get("stdout") or "").strip()

    def run(self, sdk, args):
        """Discover the installation, pull its upstream, and activate a build."""
        if sdk.paths.get("platform") != "darwin":
            return "UI deployment currently supports the macOS installation only."

        sdk.ui.progress("Locating the installed UI repository...")
        root = self._run(sdk, ["/bin/sh", "-c", '''set -eu
ui_plist="$HOME/Library/LaunchAgents/com.secondbrain.ui.plist"
if [ ! -f "$ui_plist" ]; then
    printf 'UI installation not found: %s\\n' "$ui_plist" >&2
    exit 1
fi
ui_repo=$(/usr/libexec/PlistBuddy -c 'Print :WorkingDirectory' "$ui_plist")
cd "$ui_repo"
test -f deploy/macos/manage.sh && test -f package.json
pwd -P
'''], "Installation discovery")
        if not root.startswith("/") or "\n" in root:
            raise RuntimeError("The UI installation did not identify an absolute checkout path.")

        sdk.ui.progress("Pulling the UI repository...")
        pulled = self._run(sdk, ["git", "pull", "--ff-only"], "Repository pull", cwd=root, timeout=120)

        sdk.ui.progress("Repository pulled. Issuing sh deploy/macos/manage.sh update...")
        deployed = self._run(sdk, ["/bin/sh", "-c", '''set -eu
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec sh deploy/macos/manage.sh update
'''], "UI deployment", cwd=root, timeout=600)
        # Indented code keeps arbitrary command output literal in Markdown.
        output = "\n".join("    " + line for line in (pulled + "\n" + deployed)[-12000:].splitlines())
        return "UI updated successfully. Reload the web app to use the new version.\n\n" + output
