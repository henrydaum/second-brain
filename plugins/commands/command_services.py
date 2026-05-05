from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_services


class ServicesCommand(BaseCommand):
    name = "services"
    description = "List registered services"
    category = "System"

    def run(self, _args, context):
        return format_services([
            {"name": name, "loaded": getattr(svc, "loaded", False), "model_name": getattr(svc, "model_name", "")}
            for name, svc in sorted((context.services or {}).items())
        ])
