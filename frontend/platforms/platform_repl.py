import logging

from frontend.platforms.base import BasePlatformAdapter
from frontend.types import FrontendAction, FrontendSession, PlatformCapabilities

logger = logging.getLogger("ReplFrontend")


class ReplPlatformAdapter(BasePlatformAdapter):
    name = "repl"
    capabilities = PlatformCapabilities(
        supports_proactive_push=True,
        max_message_chars=None,
    )

    def start(self):
        try:
            from frontend.repl.repl import run_repl
            return run_repl(
                self.ctrl, self.shutdown_fn, self.shutdown_event, self.tool_registry,
                self.services, self.config, self.root_dir, runtime=self.runtime, adapter=self,
            )
        except ImportError:
            logger.warning("REPL frontend not available — skipping.")
        except Exception as e:
            logger.error(f"REPL frontend crashed: {e}")

    def send_action(self, session: FrontendSession, action: FrontendAction):
        text = action.text.strip()
        if text:
            print(f"\n{text}", flush=True)
        if action.type == "send_attachments" and action.attachments:
            print(f"  [{len(action.attachments)} attachment(s)]", flush=True)
            for path in action.attachments:
                print(f"    • {path}", flush=True)

    def default_session(self) -> FrontendSession:
        return FrontendSession(platform=self.name, user_id="local", chat_id="console")
