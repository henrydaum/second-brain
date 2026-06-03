from plugins.BaseTool import BaseTool, ToolResult
from .helpers.echo_format import format_echo


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text through a package-installed helper."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo."}
        },
    }

    def run(self, context, text: str = ""):
        return ToolResult(llm_summary=format_echo(text), data={"text": text})
