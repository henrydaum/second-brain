"""Raw bytes crossing the boundary.

``fs.read``/``fs.write`` decode as UTF-8 with replacement, which is right for
text and silently destructive for anything else — a JPEG round-tripped through
them is no longer a JPEG. These two Requests are the honest path, and the
claim worth pinning is that the base64 the wire needs is invisible to the
plugin: it hands over ``bytes`` and gets ``bytes`` back, in either runner.
"""

from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Sandbox
from sandbox.guest import requests as R
from sandbox.policy import Chain, classify

# Bytes that are not valid UTF-8 and contain a NUL — the two things that
# would survive a text round trip visibly mangled rather than not at all.
PAYLOAD = bytes([0x89, 0x50, 0x4E, 0x47, 0x00, 0xFF, 0xFE, 0x0D, 0x0A, 0x1A])

ROUND_TRIP = '''
"""Reads bytes, writes them back, and reports what it saw."""

ISOLATION

from guest.bases import BaseTool


class Copy(BaseTool):
    """Copy a file byte for byte."""

    name = "copy"
    description = "Round-trip a file through read_bytes/write_bytes."

    def run(self, sdk, src, dst):
        """Copy and report the length and the first byte."""
        data = sdk.fs.read_bytes(src)
        sdk.fs.write_bytes(dst, data)
        return {"length": len(data), "first": data[0], "is_bytes": isinstance(data, bytes)}
'''


@pytest.fixture
def box():
    """A sandbox torn down even if a test fails."""
    made = Sandbox()
    yield made
    made.shutdown()


@pytest.fixture
def tool(tmp_path, request):
    """The round-tripping tool, optionally subprocess-isolated."""
    isolation = getattr(request, "param", "")
    source = ROUND_TRIP.replace(
        "ISOLATION", f'isolation = "{isolation}"' if isolation else "")
    path = tmp_path / "tool_copy.py"
    path.write_text(source, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────
# The round trip.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", ["", "subprocess"], indirect=True)
def test_bytes_survive_the_crossing_unchanged(box, tool, tmp_path):
    """The claim: what went in is what came out, in either runner."""
    import tempfile

    src = tmp_path / "in.png"
    # Reads are safe anywhere; writes are judged by where, so the copy has to
    # land in scratch or the run is (correctly) denied.
    dst = Path(tempfile.gettempdir()) / "sb-test-copy.png"
    dst.unlink(missing_ok=True)
    src.write_bytes(PAYLOAD)

    result = box.run(tool, "Copy", kwargs={"src": str(src), "dst": str(dst)})

    assert result.ok, result.error
    assert result.data["length"] == len(PAYLOAD)
    assert result.data["first"] == PAYLOAD[0]
    # The guest sees bytes, never the base64 the wire carried.
    assert result.data["is_bytes"] is True
    assert dst.read_bytes() == PAYLOAD


def test_text_read_would_have_corrupted_it(tmp_path):
    """Why the Request exists at all, stated as a test rather than a comment."""
    path = tmp_path / "in.png"
    path.write_bytes(PAYLOAD)
    mangled = path.read_text(encoding="utf-8", errors="replace")
    assert mangled.encode("utf-8", errors="replace") != PAYLOAD


# ──────────────────────────────────────────────────────────────────────
# Policy: the encoding must not be a way around the rule.
# ──────────────────────────────────────────────────────────────────────

def test_reading_bytes_is_as_safe_as_reading_text():
    """Both are reads, and reads are safe because egress is not."""
    request = R.Request(R.FS_READ_BYTES, {"path": "x"})
    assert classify(request, Chain()).safe
    assert request.read_only


def test_writing_bytes_is_judged_by_where_exactly_like_text(tmp_path):
    """Same act, different encoding — so it must get the same answer."""
    import tempfile
    scratch = f"{tempfile.gettempdir()}/sb-test-bytes.bin"
    assert classify(R.Request(R.FS_WRITE_BYTES, {"path": scratch, "data": "eA=="}),
                    Chain()).safe
    assert not classify(
        R.Request(R.FS_WRITE_BYTES, {"path": "main.pyw", "data": "eA=="}),
        Chain()).safe


# ──────────────────────────────────────────────────────────────────────
# Handler edges.
# ──────────────────────────────────────────────────────────────────────

def test_junk_base64_is_refused_rather_than_truncated(tmp_path):
    """Silently dropping invalid characters would write a short file."""
    from sandbox.handlers.fs_net import _fs_write_bytes

    target = tmp_path / "out.bin"
    result = _fs_write_bytes(None, {"path": str(target), "data": "not!base64!"})

    assert not result.ok
    assert "base64" in result.error
    assert not target.exists()


def test_an_oversized_file_is_refused(tmp_path, monkeypatch):
    """The cap bounds one wire frame, and base64 inflates by 4/3."""
    from sandbox.handlers import fs_net

    monkeypatch.setattr(fs_net, "MAX_READ_BINARY", 4)
    path = tmp_path / "big.bin"
    path.write_bytes(b"12345")

    result = fs_net._fs_read_bytes(None, {"path": str(path)})

    assert not result.ok
    assert "exceeds" in result.error


def test_append_mode_adds_rather_than_replaces(tmp_path):
    """``mode="append"`` opens 'ab', so a second write extends the file."""
    from sandbox.handlers.fs_net import _fs_write_bytes
    import base64

    target = tmp_path / "log.bin"
    for chunk in (b"\x00\x01", b"\x02\x03"):
        _fs_write_bytes(None, {"path": str(target),
                               "data": base64.b64encode(chunk).decode(),
                               "mode": "append"})

    assert target.read_bytes() == b"\x00\x01\x02\x03"
