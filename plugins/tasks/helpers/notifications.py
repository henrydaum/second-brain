NOTIFICATION_MODES = ("all", "important", "off")
DEFAULT_NOTIFICATION_MODE = "all"


def notification_mode(value, default=DEFAULT_NOTIFICATION_MODE) -> str:
    mode = str(value or default).strip().lower()
    return mode if mode in NOTIFICATION_MODES else default
