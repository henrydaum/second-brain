def is_main_conversation(context) -> bool:
    runtime, key = getattr(context, "runtime", None), getattr(context, "session_key", None)
    session = getattr(runtime, "sessions", {}).get(key) if runtime and key else None
    cid = getattr(session, "conversation_id", None)
    db = getattr(context, "db", None) or getattr(runtime, "db", None)
    row = db.get_conversation(cid) if db and cid else None
    return ((row or {}).get("category") or "").strip() in {"", "Main"}
