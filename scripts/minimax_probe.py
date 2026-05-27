"""Minimal MiniMax probe to isolate error 2013.

Run each test in order. The first one that 400s tells us which payload
ingredient MiniMax is rejecting.
"""
from __future__ import annotations

import json
import os
import sys

import httpx

BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
API_KEY = "sk-cp-fww4r26DB-5l1gPDwDZ5BoPlI_kheEkTK4YpkJHd1FZSF8tYF62tzWQdwLLUk86PW7TQSMhRsI56P3HJlY9Zv_y6TjzbKsRQhN74tlnNXDST6mtpud9eISE"
MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")

if not API_KEY:
    sys.exit("Set MINIMAX_API_KEY env var.")

URL = f"{BASE_URL.rstrip('/')}/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def call(label: str, payload: dict) -> None:
    print(f"\n=== {label} ===")
    try:
        r = httpx.post(URL, headers=HEADERS, json=payload, timeout=60.0)
        print(f"HTTP {r.status_code}")
        print(r.text[:800])
    except Exception as e:
        print(f"EXC: {e!r}")


# 1. Bare minimum.
call("bare", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "say hi"}],
})

# 2. Add a single system message at head.
call("one system at head", {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "say hi"},
    ],
})

# 3. Three system messages stacked at head (current Codex shape).
call("three systems at head", {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "[STATIC] You are a test."},
        {"role": "system", "content": "[SEMI] tools listed here"},
        {"role": "system", "content": "[DYNAMIC] runtime context"},
        {"role": "user", "content": "say hi"},
    ],
})

# 4. System message mid-conversation (original shape).
call("system mid-stream", {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "[STATIC] You are a test."},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "okay"},
        {"role": "system", "content": "[DYNAMIC] runtime context"},
        {"role": "user", "content": "say hi"},
    ],
})

# 5. With a tool schema attached.
call("with tools", {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "say hi"},
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "echo back",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }],
})

print("\nDone. The first FAILED test pinpoints the trigger.")
