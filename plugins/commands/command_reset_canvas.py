"""Slash command for clearing the demo canvas."""

from __future__ import annotations

from plugins.BaseCommand import BaseCommand
from plugins.tools.helpers import layered_canvas as lc


class ResetCanvasCommand(BaseCommand):
    name = "reset_canvas"
    description = "Clear the current canvas image and skill chain."
    category = "Conversation"

    def run(self, args, context):
        lc.reset(getattr(context, "session_key", None) or "local")
        return "Canvas reset."
