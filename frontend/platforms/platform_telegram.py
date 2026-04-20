import logging

from frontend.platforms.base import BasePlatformAdapter
from frontend.types import FrontendSession, PlatformCapabilities

logger = logging.getLogger("TelegramFrontend")


class TelegramPlatformAdapter(BasePlatformAdapter):
    name = "telegram"
    capabilities = PlatformCapabilities(
        supports_typing=True,
        supports_buttons=True,
        supports_message_edit=True,
        supports_attachments_in=True,
        supports_attachments_out=True,
        supports_inline_forms=True,
        supports_proactive_push=True,
        supports_rich_text=True,
        max_message_chars=4096,
        max_upload_size=50 * 1024 * 1024,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sender = None

    def start(self):
        try:
            from frontend.telegram.telegram import run_telegram_bot
            return run_telegram_bot(
                self.ctrl, self.shutdown_fn, self.shutdown_event, self.tool_registry,
                self.services, self.config, self.root_dir, runtime=self.runtime, adapter=self,
            )
        except ImportError:
            logger.warning("Telegram frontend not available — skipping.")
        except Exception as e:
            logger.error(f"Telegram frontend crashed: {e}")

    def set_sender(self, sender):
        self._sender = sender

    def send_action(self, session, action):
        if self._sender:
            self._sender(session, action)

    def default_session(self) -> FrontendSession | None:
        user_id = str(int(self.config.get("telegram_allowed_user_id", 0) or 0))
        if user_id == "0":
            return None
        return FrontendSession(platform=self.name, user_id=user_id, chat_id=user_id)
