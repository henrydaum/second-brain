from plugins.BaseCommand import BaseCommand


class HelpCommand(BaseCommand):
    name = "help"
    description = "List available commands"
    category = "Conversation"

    def run(self, _args, context):
        registry = getattr(context, "command_registry", None)
        if registry is None:
            return "No command registry is available."
        return registry.help_text()
