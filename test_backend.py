"""
Quick smoke test for the backend WebSocket server.

Run this while Second Brain is running (or standalone with a mock).
It connects, creates a session, sends a /services command, and prints
all received events.

Usage:
    python test_backend.py
    python test_backend.py --chat "What files do I have?"
"""

import argparse
import asyncio
import json
import sys

import websockets


async def main(port: int, chat_message: str | None):
    uri = f"ws://127.0.0.1:{port}"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri) as ws:
        # 1. Create a session
        await ws.send(json.dumps({
            "type": "session.create",
            "request_id": "test-001",
        }))

        resp = json.loads(await ws.recv())
        print(f"\n[RECV] {resp['type']}")
        print(json.dumps(resp, indent=2))

        session_id = resp.get("session_id")
        if not session_id:
            print("ERROR: No session_id in response")
            return

        # 2. Send a slash command
        print("\n--- Sending /services command ---")
        await ws.send(json.dumps({
            "type": "command.send",
            "request_id": "test-002",
            "session_id": session_id,
            "command": "services",
            "arg": "",
        }))

        resp = json.loads(await ws.recv())
        print(f"\n[RECV] {resp['type']}")
        print(json.dumps(resp, indent=2))

        # 3. Optionally send a chat message
        if chat_message:
            print(f"\n--- Sending chat: {chat_message!r} ---")
            await ws.send(json.dumps({
                "type": "chat.send",
                "request_id": "test-003",
                "session_id": session_id,
                "message": chat_message,
            }))

            # Collect events until agent.done or agent.error
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                resp = json.loads(raw)
                print(f"\n[RECV] {resp['type']}")
                if resp["type"] in ("agent.done", "agent.error", "agent.cancelled"):
                    print(json.dumps(resp, indent=2))
                    break
                elif resp["type"] == "agent.message":
                    role = resp.get("role", "")
                    content = resp.get("content", "")[:200]
                    print(f"  role={role}  content={content!r}")
                elif resp["type"] == "agent.tool_result":
                    print(f"  tool={resp.get('tool_name')}  success={resp.get('success')}")
                else:
                    print(json.dumps(resp, indent=2))

        # 4. Destroy session
        await ws.send(json.dumps({
            "type": "session.destroy",
            "request_id": "test-004",
            "session_id": session_id,
        }))

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the backend WebSocket server")
    parser.add_argument("--port", type=int, default=5150, help="WebSocket port (default 5150)")
    parser.add_argument("--chat", type=str, default=None, help="Optional chat message to send")
    args = parser.parse_args()
    asyncio.run(main(args.port, args.chat))
