import base64
from email import message_from_bytes

from plugins.services.gmailService import GmailService


def _b64(text):
    return base64.urlsafe_b64encode(text.encode()).decode()


def test_gmail_parse_message_reads_nested_multipart_body():
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


class _Client:
    def __init__(self, sent):
        self.sent = sent

    def users(self):
        return self

    def messages(self):
        return self

    def send(self, userId, body):
        self.sent["userId"] = userId
        self.sent["body"] = body
        return self

    def execute(self):
        return {"id": "sent-1"}
