"""Real-LLM compaction stress — a path neither fuzzer reaches.

The fuzzer's fake LLM sets ``context_size=0`` to disable proactive compaction, so
the compactor service (which calls the real model to summarize and rewrites
history *in place*) is never exercised end to end. This drives a real MiniMax
conversation with a deliberately tiny ``context_size`` so compaction fires, then
checks that (a) it actually fired, (b) the kernel stays invariant-clean across
it, and (c) a fact planted *before* compaction survives in the summary.

Run: ``python -m stress.stress_compaction``
"""

from __future__ import annotations

from pathlib import Path

from config import config_manager
from stress.boot import boot_kernel
from stress.driver import _build_minimax_llm, DRIVER_DATA_DIR, MINIMAX_PROFILE
from stress.invariants import check_invariants

SESSION = "compaction"
CODEWORD = "ZEPHYR-Neptune-7741"


def main():
    real = config_manager.load()
    DRIVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    llm = _build_minimax_llm(real, DRIVER_DATA_DIR, MINIMAX_PROFILE)
    llm.context_size = 1500  # force proactive compaction at ~1200 prompt tokens

    kernel = boot_kernel(llm=llm)
    notices: list[str] = []
    kernel.runtime.on_notice = notices.append
    # boot skips autoload when an LLM is injected; the compactor must be loaded.
    kernel.services["compactor"].load()

    try:
        rt = kernel.runtime
        rt.set_session_user(SESSION, 1)
        rt.active_session_key = SESSION

        # Plant a fact, then pile on turns to grow the prompt past the threshold.
        prompts = [
            f"Please remember this codeword exactly: {CODEWORD}. Acknowledge it.",
            "Tell me a few sentences about the history of cartography.",
            "Now a few sentences about deep-sea exploration.",
            "And a few sentences about the development of the printing press.",
            "Summarize what we've discussed so far in two sentences.",
            "What was the exact codeword I gave you at the very start?",
        ]
        any_violation = False
        compacted = False
        for i, p in enumerate(prompts, 1):
            result = rt.iterate_agent_turn(SESSION, p)
            reply = "\n".join(m for m in result.messages if m).strip()
            v = check_invariants(kernel)
            if v:
                any_violation = True
                print(f"  turn {i}: INVARIANT VIOLATIONS: {[str(x) for x in v]}")
            if any("Compact" in n for n in notices) and not compacted:
                compacted = True
                print(f"  turn {i}: compaction fired ({[n for n in notices if 'Compact' in n]})")
            print(f"  turn {i} ({len(rt.sessions[SESSION].history)} msgs in history): {reply[:90]}")

        sess = rt.sessions[SESSION]
        print("\n--- results ---")
        print("compaction fired      :", compacted)
        print("has_compaction_checkpoint:", sess.has_compaction_checkpoint)
        print("final history length  :", len(sess.history))
        recalled = CODEWORD in (sess.history[-1].get("content") or "")
        print("codeword recalled post-compaction:", recalled)
        print("invariants clean throughout:", not any_violation)
        if not compacted:
            print("NOTE: compaction did not trigger — provider prompt_tokens may be below "
                  "0.8*context_size; lower context_size and re-run.")
    finally:
        kernel.close()


if __name__ == "__main__":
    main()
