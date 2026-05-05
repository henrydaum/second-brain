from plugins.BaseCommand import BaseCommand


class FrontendsCommand(BaseCommand):
    name = "frontends"
    description = "List enabled frontends"
    category = "System"

    def run(self, _args, context):
        names = sorted((context.config or {}).get("enabled_frontends", ["repl", "telegram"]))
        return "Frontends:\n" + "\n".join(f"  {name}" for name in names) if names else "No frontends enabled."
