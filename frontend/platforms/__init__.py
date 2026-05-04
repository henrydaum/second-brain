from frontend.platforms.base import BasePlatformAdapter
from frontend.platforms.bootstrap import start_frontends
from frontend.platforms.platform_telegram import TelegramPlatformAdapter

__all__ = [
    "BasePlatformAdapter",
    "TelegramPlatformAdapter",
    "start_frontends",
]
