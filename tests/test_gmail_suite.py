"""Regression tests for Gmail suite."""

import base64
from email import message_from_bytes
from types import SimpleNamespace

from plugins.services.gmailService import GmailService
from plugins.tools.tool_email_check import EmailCheck


def _b64(text):
    """Internal helper to handle b64."""
    return base64.urlsafe_b64encode(text.encode()).decode()


def test_gmail_parse_message_reads_nested_multipart_body():
    """Verify Gmail parse message reads nested multipart body."""
    msg = {
        "id": "m1",
        "threadId": "t1",
        "internalDate": "0",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Hello"},
                {"name": "From", "value": "Professor <p@example.edu>"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Message-ID", "value": "<orig@example.edu>"},
            ],
            "parts": [{
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64("plain body")}},
                    {"mimeType": "text/html", "body": {"data": _b64("<b>html body</b>")}},
                ],
            }],
        },
    }

    parsed = GmailService()._parse_message(msg)

    assert parsed["body_plain"] == "plain body"
    assert parsed["body_html"] == "<b>html body</b>"
    assert parsed["message_id_header"] == "<orig@example.edu>"


def test_gmail_reply_uses_original_message_id_headers():
    """Verify Gmail reply uses original message ID headers."""
    svc = GmailService()
    sent = {}
    svc.get_message = lambda _: {
        "sender": "Professor <p@example.edu>",
        "thread_id": "thread-1",
        "subject": "Office hours",
        "message_id_header": "<orig@example.edu>",
        "references": "<root@example.edu>",
    }
    svc.get_client = lambda: _Client(sent)

    assert svc.reply_to("gmail-internal-id", "Thanks", from_address="me@example.com") == "sent-1"

    raw = sent["body"]["raw"]
    mime = message_from_bytes(base64.urlsafe_b64decode(raw.encode()))
    assert sent["body"]["threadId"] == "thread-1"
    assert mime["To"] == "p@example.edu"
    assert mime["In-Reply-To"] == "<orig@example.edu>"
    assert mime["References"] == "<root@example.edu> <orig@example.edu>"


def test_email_tools_use_main_conversation_instead_of_is_subagent():
    """Verify email tools use main conversation instead of is subagent."""
    gmail = SimpleNamespace(
        loaded=True,
        fetch_inbox=lambda max_results: [],
        search=lambda *_, **__: [],
        get_message=lambda *_: None,
    )
    db = _Db({1: {"id": 1, "category": None}, 2: {"id": 2, "category": "Scheduled"}})
    runtime = SimpleNamespace(sessions={"chat": SimpleNamespace(conversation_id=1), "job": SimpleNamespace(conversation_id=2)})

    main = SimpleNamespace(db=db, runtime=runtime, session_key="chat", services={"gmail": gmail}, config={})
    scheduled = SimpleNamespace(db=db, runtime=runtime, session_key="job", services={"gmail": gmail}, config={})

    assert EmailCheck().run(main, scope="inbox").success
    result = EmailCheck().run(scheduled, scope="inbox")
    assert not result.success
    assert "Non-main conversation" in result.error


class _Client:
    """Client."""
    def __init__(self, sent):
        """Initialize the client."""
        self.sent = sent

    def users(self):
        """Handle users."""
        return self

    def messages(self):
        """Handle messages."""
        return self

    def send(self, userId, body):
        """Send client."""
        self.sent["userId"] = userId
        self.sent["body"] = body
        return self

    def execute(self):
        """Handle execute."""
        return {"id": "sent-1"}


class _Db:
    """DB."""
    def __init__(self, rows):
        """Initialize the DB."""
        self.rows = rows

    def get_conversation(self, conversation_id):
        """Get conversation."""
        return self.rows.get(conversation_id)
