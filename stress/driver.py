"""Claude+MiniMax adversarial driver — an intelligent human at the REPL.

This is the "wire myself in as a user" harness. A blind fuzzer is great at
ordering bugs; it is hopeless at *semantic* exploration ("install a parser
mid-conversation, hand it a weird file, then cancel a form halfway"). That kind
of probing wants a reasoning agent in the user seat. So: **MiniMax is the
agent's brain**, and the operator (Claude, via the shell — or you) is the
adversarial user, deciding each next message after reading the last reply.

The trick that makes turn-by-turn shell driving possible is *persistence*: the
kernel boots against a fixed data dir (``.stress_driver/`` by default) with a
real SQLite DB, so each ``say`` invocation restores the last-active conversation,
drives one turn, prints the reply, persists, and exits. State survives across
processes — which itself dogfoods the kernel's restore/persistence path.

The MiniMax backend is wired with **zero global footprint**: the ``service-litellm``
backend source is fetched from the ``origin/store`` git ref, written under the
driver's data dir, imported, and instantiated from your real MiniMax profile in
``config.json``. Nothing is installed into the global plugin tree.

Usage::

    python -m stress.driver say "hi — what can you do?"
    python -m stress.driver say "/commands"
    python -m stress.driver history
    python -m stress.driver check      # run the invariant oracle on live state
    python -m stress.driver reset      # start a fresh conversation
    python -m stress.driver info       # show backend / model / data dir
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from config import config_manager
from stress.boot import boot_kernel, _ROOT
from stress.invariants import check_invariants

DRIVER_DATA_DIR = _ROOT / ".stress_driver"
SESSION_KEY = "driver"
MINIMAX_PROFILE = "minimax/MiniMax-M2.7"  # default; override with --profile


# ── MiniMax backend wiring (no global install) ─────────────────────────

def _fetch_litellm_backend(dest_dir: Path):
    """Materialise the store's LiteLLM backend and return its class."""
    dest = dest_dir / "_stress_litellm.py"
    if not dest.exists():
        src = subprocess.check_output(
            ["git", "show", "origin/store:packages/service-litellm/files/services/service_litellm.py"],
            cwd=str(_ROOT),
        )
        dest.write_bytes(src)
    spec = importlib.util.spec_from_file_location("_stress_litellm", dest)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_stress_litellm"] = module
    spec.loader.exec_module(module)
    return module.LiteLLMService


def _build_minimax_llm(real_config: dict, dest_dir: Path, profile_name: str):
    """Build a loaded LiteLLM backend from the on-disk MiniMax profile."""
    profiles = real_config.get("llm_profiles") or {}
    profile = profiles.get(profile_name)
    if profile is None:
        raise SystemExit(
            f"No LLM profile {profile_name!r} found in config.json. "
            f"Available: {sorted(profiles)}"
        )
    import os
    cls = _fetch_litellm_backend(dest_dir)
    api_key = profile.get("llm_api_key", "") or ""
    resolved_key = os.environ.get(api_key, api_key) if api_key else None
    base_url = profile.get("llm_endpoint") or None
    llm = cls(profile_name, api_key=resolved_key, base_url=base_url)
    llm.capabilities.update({
        k: v for k, v in (profile.get("llm_capabilities") or {}).items()
        if k in llm.capabilities
    })
    ctx = int(profile.get("llm_context_size", 0) or 0)
    if ctx > 0:
        llm.context_size = ctx
    if not llm.load():
        raise SystemExit("LiteLLM backend failed to load (check API key / network).")
    return llm


# ── kernel lifecycle ────────────────────────────────────────────────────

def _boot(profile_name: str):
    real = config_manager.load()
    DRIVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    llm = _build_minimax_llm(real, DRIVER_DATA_DIR, profile_name)
    kernel = boot_kernel(
        llm=llm,
        data_dir=str(DRIVER_DATA_DIR),
        # Keep the agent profile dict from real config so /agent etc. behave,
        # but route the LLM straight to our injected MiniMax backend.
        config_overrides={"agent_profiles": real.get("agent_profiles", {"default": {"llm": "default"}})},
    )
    return kernel


_ACTIVE_PTR = DRIVER_DATA_DIR / "active_conversation"


def _restore_or_new(kernel):
    """Bind the base user and restore (or create) the driver conversation.

    Continuity across separate process invocations is owned explicitly via a
    pointer file rather than the per-user ``last_active_conversation_id`` blob
    (which only gets persisted by a frontend that sets ``active_session_key``).
    Marking the session active + attended makes it behave like a real REPL
    session (interactive-tool gating, notify suppression).
    """
    rt = kernel.runtime
    rt.set_session_user(SESSION_KEY, 1)
    rt.active_session_key = SESSION_KEY
    cid = None
    if _ACTIVE_PTR.exists():
        try:
            cid = int(_ACTIVE_PTR.read_text().strip())
        except Exception:
            cid = None
        if cid is not None and kernel.db.get_conversation(cid) is None:
            cid = None  # pointer went stale (DB reset)
    if cid is not None:
        rt.load_conversation(SESSION_KEY, cid, override=True)
    else:
        # new_conversation binds lazily — the row is created on first message,
        # so the pointer is saved post-turn by _save_pointer(), not here.
        rt.new_conversation(SESSION_KEY)
    rt.set_session_attended(SESSION_KEY, True)


def _save_pointer(kernel) -> None:
    sess = kernel.runtime.sessions.get(SESSION_KEY)
    if sess and sess.conversation_id is not None:
        _ACTIVE_PTR.write_text(str(sess.conversation_id))


# ── rendering ─────────────────────────────────────────────────────────

def _render_turn(result) -> None:
    data = getattr(result, "data", {}) or {}
    for msg in data.get("new_messages") or []:
        role = msg.get("role")
        if role == "tool":
            name = msg.get("name", "tool")
            content = str(msg.get("content", ""))[:300]
            print(f"  ⚙ tool[{name}] -> {content}")
        elif role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", tc)
                print(f"  → calls {fn.get('name')}({fn.get('arguments', '')[:120]})")
    final = "\n".join(m for m in result.messages if m).strip()
    print(f"\nMiniMax: {final or '(no text)'}")


# ── subcommands ───────────────────────────────────────────────────────

def cmd_say(args):
    kernel = _boot(args.profile)
    try:
        _restore_or_new(kernel)
        result = kernel.runtime.iterate_agent_turn(SESSION_KEY, args.message)
        _save_pointer(kernel)
        _render_turn(result)
        violations = check_invariants(kernel)
        if violations:
            print("\n‼ INVARIANT VIOLATIONS:")
            for v in violations:
                print(f"   {v}")
        else:
            print("\n✓ invariants clean")
    finally:
        kernel.close()


def cmd_history(args):
    kernel = _boot(args.profile)
    try:
        _restore_or_new(kernel)
        sess = kernel.runtime.sessions.get(SESSION_KEY)
        if not sess or sess.conversation_id is None:
            print("(no conversation)")
            return
        for msg in kernel.db.get_conversation_messages(sess.conversation_id):
            role = msg["role"]
            content = str(msg["content"])[:200]
            print(f"[{role}] {content}")
    finally:
        kernel.close()


def cmd_check(args):
    kernel = _boot(args.profile)
    try:
        _restore_or_new(kernel)
        violations = check_invariants(kernel)
        if violations:
            for v in violations:
                print(v)
            raise SystemExit(1)
        print("✓ invariants clean")
    finally:
        kernel.close()


def cmd_reset(args):
    kernel = _boot(args.profile)
    try:
        kernel.runtime.set_session_user(SESSION_KEY, 1)
        kernel.runtime.new_conversation(SESSION_KEY)
        sess = kernel.runtime.sessions.get(SESSION_KEY)
        if sess and sess.conversation_id is not None:
            _ACTIVE_PTR.write_text(str(sess.conversation_id))
        print("Started a fresh conversation.")
    finally:
        kernel.close()


def cmd_info(args):
    real = config_manager.load()
    print(f"data dir : {DRIVER_DATA_DIR}")
    print(f"profile  : {args.profile}")
    print(f"available: {sorted((real.get('llm_profiles') or {}))}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stress.driver", description=__doc__)
    parser.add_argument("--profile", default=MINIMAX_PROFILE, help="LLM profile name from config.json")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("say", help="send one user message and drive a turn")
    s.add_argument("message")
    s.set_defaults(func=cmd_say)
    sub.add_parser("history", help="print the current conversation").set_defaults(func=cmd_history)
    sub.add_parser("check", help="run the invariant oracle").set_defaults(func=cmd_check)
    sub.add_parser("reset", help="start a fresh conversation").set_defaults(func=cmd_reset)
    sub.add_parser("info", help="show backend/model/data dir").set_defaults(func=cmd_info)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
