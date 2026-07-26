"""The guest's transport — how sandboxed code reaches the kernel.

The guest has exactly one way out: write a Request, block, read a Result. No
queues, no threads, no interpreter. Whatever is on the other end of the two
streams — a pipe today, a container socket later — is not the guest's concern,
which is what makes the same plugin code run unchanged under every runner.

The host has its own in-process channel (see ``sandbox.interpreter``). Both
satisfy the same two-method shape: ``send`` and ``log``.
"""

from __future__ import annotations

from . import protocol
from .requests import Request, Result


class Terminated(BaseException):
    """Raised by ``sdk.respond`` to unwind the sandboxed code.

    Deriving from ``BaseException`` rather than ``Exception`` on purpose, for
    the same reason ``KeyboardInterrupt`` does: a plugin wrapping its work in
    ``except Exception`` must not accidentally swallow the kernel tearing it
    down and carry on running.

    Sandboxed code has to ask to end its own life, and asking has to actually
    end it — otherwise ``respond`` would record an answer and then keep
    running. Every runner catches this and takes the carried value as the
    execution's result.
    """

    def __init__(self, value):
        super().__init__("sandboxed code responded")
        self.value = value


class PipeChannel:
    """Newline-delimited JSON over two byte streams.

    Takes any reader/writer pair, not pipes specifically — a container's
    socket or attach stream works here unchanged.
    """

    def __init__(self, reader, writer):
        self._reader = reader
        self._writer = writer

    def send(self, request: Request) -> Result:
        """Send a Request upstream and block for the answer."""
        protocol.write_message(self._writer, {
            "kind": protocol.REQUEST,
            "request": request.to_dict(),
        })
        message = protocol.read_message(self._reader)
        if message is None:
            # The host closed the channel: we are being torn down. Unwind
            # rather than spin making Requests nobody will answer.
            raise Terminated(None)
        if message.get("kind") != protocol.RESULT:
            raise protocol.ProtocolError(
                f"expected a result, got {message.get('kind')}")
        return Result.from_dict(message["result"])

    def log(self, level: str, message: str) -> None:
        """Send a log line upstream to the kernel's sink."""
        protocol.write_message(self._writer, {
            "kind": protocol.LOG, "level": level, "message": message,
        })
