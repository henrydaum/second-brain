from plugins.BaseCommand import BaseCommand


class NewCommand(BaseCommand):
    name = "new"
    description = "Start a new conversation"
    category = "Conversation"

    def run(self, args, context):
        runtime = getattr(context, "runtime", None)
        if runtime and getattr(context, "session_key", None):
            return "\n".join(runtime.new_conversation(context.session_key).messages)
        return f"New conversation started. Agent: {(getattr(context, 'config', {}) or {}).get('active_agent_profile') or 'default'}."
