from plugins.BaseCommand import BaseCommand


class CommandsCommand(BaseCommand):
    name = "commands"
    description = "List available commands"
    category = "Conversation"

    def run(self, _args, context):
        registry = getattr(context, "command_registry", None)
        return registry.help_text() if registry else "No command registry is available."
