"""Kernel stress-testing harness.

This package is **test/QA scaffolding**, not part of the kernel. It is the
"syzkaller-for-an-agent-kernel" toolkit referred to in the testing notes:

- ``boot``        — boot a *full* kernel headlessly against a throwaway data dir,
                    with a pluggable LLM (network-free fake or the real backend).
- ``fake_llm``    — deterministic / random ("monkey") LLM backends so the agent
                    turn can be driven thousands of times with no network.
- ``invariants``  — the "sanitizer" layer: assert kernel invariants after every
                    operation (no leaked sessions/threads, DB integrity, user
                    ownership never crosses, registries consistent).
- ``fuzz_runtime``— a ``hypothesis`` stateful machine that throws random *valid
                    sequences* of lifecycle operations at the runtime and checks
                    the invariants after each step.
- ``driver``      — the Claude+MiniMax adversarial driver: an intelligent human
                    sitting at the REPL, exploring flows a blind fuzzer won't.

Nothing here is imported by the kernel; it only ever imports *from* it.
"""
