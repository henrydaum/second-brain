"""
Gmail Service.

Handles OAuth 2.0 authentication and provides the Gmail API service object.
Follows the standard service interface so it integrates with the services dict.

Usage:
    gmail = context.services.get("gmail")
    gmail.load()   # opens browser for OAuth if no valid token
    client = gmail.get_client()
    messages = gmail.fetch_inbox(max_results=20)
    message  = gmail.get_message(message_id)
    gmail.send_message(to, subject, body)
    gmail.mark_read(message_id)
    gmail.unload()

Credentials (stored in DATA_DIR):
    - credentials.json  — OAuth client secret (user provides once)
    - gmail_token.json  — OAuth refresh token (auto-generated)

Scope: https://www.googleapis.com/auth/gmail.modify
(read, send, modify labels — needed to mark as read/unread)
"""

import logging
from plugins.BaseService import BaseService
from paths import DATA_DIR

logger = logging.getLogger("GmailService")


class GmailService(BaseService):
    """Gmail service."""
    name = "gmail"
    model_name = "gmail"
    shared = False  # get_client() returns a fresh HTTP transport per call

    def __init__(self):
        """Initialize the Gmail service."""
        super().__init__()
        self._creds = None
        self.service = None  # backward compat
        self._self_address = None
        self._labels_cache: list[dict] | None = None

    def _load(self) -> bool:
        """Internal helper to load Gmail service."""
        if not self._is_connected():
            logger.warning("[Gmail] No internet — cannot authenticate.")
            return False

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            logger.error(
                "[Gmail] Missing libraries: "
                "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
            return False

        cred_path = DATA_DIR / "credentials.json"
        if not cred_path.exists():
            logger.error(f"[Gmail] No credentials.json at {cred_path}")
            return False

        SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
        token_path = DATA_DIR / "gmail_token.json"
        creds = None

        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception as e:
                logger.debug(f"Token load failed, will re-auth: {e}")

        try:
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    logger.info("[Gmail] Refreshing expired token...")
                    creds.refresh(Request())
                else:
                    logger.info("[Gmail] Opening browser for OAuth consent...")
                    flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
                    creds = flow.run_local_server(port=0)
                with open(token_path, "w") as f:
                    f.write(creds.to_json())

            self._creds = creds
            self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            self.loaded = True
            return True

        except Exception as e:
            logger.error(f"[Gmail] Authentication failed: {e}")
            return False

    def get_client(self):
        """Return a fresh Gmail API client. Thread-safe."""
        if not self.loaded or not self._creds:
            return None
        try:
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            if self._creds.expired and self._creds.refresh_token:
                self._creds.refresh(Request())
            return build("gmail", "v1", credentials=self._creds, cache_discovery=False)
        except Exception as e:
            logger.error(f"[Gmail] get_client failed: {e}")
            return None

    def unload(self):
        """Handle unload."""
        self._creds = None
        self.service = None
        self._self_address = None
        self._labels_cache = None
        self.loaded = False
        logger.info("[Gmail] Service unloaded.")

    def list_labels(self, force_refresh: bool = False) -> list[dict]:
        """Return Gmail labels as [{id, name, type}]. Cached on the instance."""
        if self._labels_cache is not None and not force_refresh:
            return self._labels_cache
        client = self.get_client()
        if not client:
            return []
        try:
            resp = client.users().labels().list(userId="me").execute()
            self._labels_cache = [
                {"id": l.get("id", ""), "name": l.get("name", ""), "type": l.get("type", "user")}
                for l in resp.get("labels", [])
            ]
            return self._labels_cache
        except Exception as e:
            logger.error(f"[Gmail] list_labels failed: {e}")
            return []

    def modify_labels(self, message_id: str, add_ids: list[str], remove_ids: list[str]) -> bool:
        """Public wrapper around _modify_labels for label add/remove operations."""
        return self._modify_labels(message_id, add_ids, remove_ids)

    def get_self_address(self) -> str:
        """Return the authenticated Google account's email address (cached)."""
        if self._self_address:
            return self._self_address
        client = self.get_client()
        if not client:
            return ""
        try:
            profile = client.users().getProfile(userId="me").execute()
            self._self_address = (profile.get("emailAddress") or "").strip()
            return self._self_address
        except Exception as e:
            logger.error(f"[Gmail] getProfile failed: {e}")
            return ""

    # ── Inbox access ──────────────────────────────────────────────────────────

    def fetch_inbox(self, max_results: int = 50, label: str = "INBOX") -> list[dict]:
        """Fetch message summaries from a label. Leaves messages UNREAD."""
        client = self.get_client()
        if not client:
            return []
        try:
            results = (
                client.users()
                .messages()
                .list(userId="me", labelIds=label, maxResults=max_results)
                .execute()
            )
            messages = results.get("messages", [])
            return [self._summarize(client.users().messages().get(
                userId="me", id=m["id"], format="metadata").execute())
                for m in messages]
        except Exception as e:
            logger.error(f"[Gmail] fetch_inbox failed: {e}")
            return []

    def search(self, query: str, max_results: int = 50) -> list[dict]:
        """Search Gmail with a raw query (e.g. 'is:unread', 'from:foo@bar')."""
        client = self.get_client()
        if not client:
            return []
        try:
            results = (
                client.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )
            messages = results.get("messages", [])
            return [self._summarize(client.users().messages().get(
                userId="me", id=m["id"], format="metadata").execute())
                for m in messages]
        except Exception as e:
            logger.error(f"[Gmail] search failed for {query!r}: {e}")
            return []

    def fetch_inbox_aliased(self, alias_address: str, max_results: int = 50) -> list[dict]:
        """Fetch messages addressed to a Gmail alias (to:alias@…)."""
        return self.search(f'to:"{alias_address}"', max_results=max_results)

    def get_message(self, message_id: str) -> dict | None:
        """Fetch full message metadata + body."""
        client = self.get_client()
        if not client:
            return None
        try:
            msg = client.users().messages().get(
                userId="me", id=message_id, format="full").execute()
            return self._parse_message(msg)
        except Exception as e:
            logger.error(f"[Gmail] get_message failed for {message_id}: {e}")
            return None

    def mark_read(self, message_id: str) -> bool:
        """Handle mark read."""
        return self._modify_labels(message_id, add=[], remove=["UNREAD"])

    def mark_unread(self, message_id: str) -> bool:
        """Handle mark unread."""
        return self._modify_labels(message_id, add=["UNREAD"], remove=[])

    # ── Sending ───────────────────────────────────────────────────────────────

    def send_message(self, to: str, subject: str, body: str,
                     cc: str = "", attachments: list[str] | None = None,
                     from_address: str = None) -> str | None:
        """Send an email. Returns sent message ID or None."""
        client = self.get_client()
        if not client:
            return None
        try:
            import base64
            import mimetypes
            import os
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.audio import MIMEAudio
            from email.mime.image import MIMEImage
            from email.utils import formatdate

            msg = MIMEMultipart()
            msg["To"] = to
            msg["Subject"] = subject
            msg["From"] = from_address or "me"
            msg["Date"] = formatdate(localtime=True)
            if cc:
                msg["Cc"] = cc
            msg.attach(MIMEText(body, "plain"))

            _attach_files(msg, attachments)

            encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            sent = client.users().messages().send(
                userId="me", body={"raw": encoded}).execute()
            logger.info(f"[Gmail] Sent {sent['id']} to {to} with {len(attachments or [])} attachment(s)")
            return sent["id"]
        except Exception as e:
            logger.error(f"[Gmail] send_message failed: {e}")
            return None

    def reply_to(self, message_id: str, body: str,
                 subject_prefix: str = "Re: ", attachments: list[str] | None = None,
                 from_address: str = None) -> str | None:
        """Reply to a message in the same thread."""
        client = self.get_client()
        if not client:
            return None
        try:
            original = self.get_message(message_id)
            if not original:
                return None

            from_email = original.get("sender", "")
            if "<" in from_email:
                from_email = from_email.split("<")[1].rstrip(">")

            thread_id = original.get("thread_id", "")
            subject = original.get("subject", "")
            if not subject.startswith("Re: ") and not subject.startswith("Fwd: "):
                subject = subject_prefix + subject

            msg_id_header = original.get("message_id_header", "") or f"<{message_id}>"
            references = original.get("references", "")

            import base64
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.utils import formatdate

            msg = MIMEMultipart()
            msg["To"] = from_email
            msg["Subject"] = subject
            msg["From"] = from_address or "me"
            msg["Date"] = formatdate(localtime=True)
            if msg_id_header:
                msg["In-Reply-To"] = msg_id_header
                msg["References"] = f"{references} {msg_id_header}".strip()
            msg.attach(MIMEText(body, "plain"))

            _attach_files(msg, attachments)

            encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            sent = client.users().messages().send(
                userId="me",
                body={"raw": encoded, "threadId": thread_id}
            ).execute()
            logger.info(f"[Gmail] Replied in thread {thread_id} with {len(attachments or [])} attachment(s)")
            return sent["id"]
        except Exception as e:
            logger.error(f"[Gmail] reply_to failed: {e}")
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_connected() -> bool:
        """Return whether connected."""
        try:
            import requests
            requests.head("https://www.google.com", timeout=3)
            return True
        except Exception:
            return False

    def _modify_labels(self, message_id: str, add: list, remove: list) -> bool:
        """Internal helper to handle modify labels."""
        client = self.get_client()
        if not client:
            return False
        try:
            client.users().messages().modify(
                userId="me", id=message_id,
                body={"addLabelIds": add, "removeLabelIds": remove}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"[Gmail] _modify_labels failed for {message_id}: {e}")
            return False

    @staticmethod
    def _header(headers: dict, name: str) -> str:
        """Internal helper to handle header."""
        return headers.get(name, "")

    def _summarize(self, msg: dict) -> dict:
        """Internal helper to handle summarize."""
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        return {
            "message_id": msg["id"],
            "thread_id": msg["threadId"],
            "subject": self._header(headers, "Subject"),
            "sender": self._header(headers, "From"),
            "received_at": int(msg.get("internalDate", 0)) / 1000.0,
            "snippet": msg.get("snippet", ""),
            "is_read": "UNREAD" not in msg.get("labelIds", []),
            "labels": msg.get("labelIds", []),
        }

    def _parse_message(self, msg: dict) -> dict:
        """Internal helper to parse message."""
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body_plain, body_html = self._body_parts(msg["payload"])

        return {
            "message_id": msg["id"],
            "thread_id": msg["threadId"],
            "subject": self._header(headers, "Subject"),
            "sender": self._header(headers, "From"),
            "recipients": self._header(headers, "To"),
            "cc": self._header(headers, "Cc"),
            "body_plain": body_plain,
            "body_html": body_html,
            "received_at": int(msg.get("internalDate", 0)) / 1000.0,
            "is_read": "UNREAD" not in msg.get("labelIds", []),
            "labels": msg.get("labelIds", []),
            "message_id_header": self._header(headers, "Message-ID"),
            "references": self._header(headers, "References"),
        }

    def _body_parts(self, part: dict) -> tuple[str, str]:
        """Internal helper to handle body parts."""
        import base64
        plain = html = ""
        data = part.get("body", {}).get("data")
        if data:
            decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            plain = decoded if part.get("mimeType") == "text/plain" else ""
            html = decoded if part.get("mimeType") == "text/html" else ""
        for child in part.get("parts", []):
            child_plain, child_html = self._body_parts(child)
            plain, html = plain or child_plain, html or child_html
        return plain, html


def _attach_files(msg, attachments):
    """Internal helper to handle attach files."""
    if not attachments:
        return
    import mimetypes
    import os
    from email.mime.audio import MIMEAudio
    from email.mime.image import MIMEImage
    from email.mime.base import MIMEBase
    import email.encoders
    for path in attachments:
        if not os.path.isfile(path):
            logger.warning(f"[Gmail] Attachment not found, skipping: {path}")
            continue
        content_type, _ = mimetypes.guess_type(path)
        if content_type is None:
            content_type = "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        filename = os.path.basename(path)
        if content_type.startswith("image/"):
            part = MIMEImage(data, name=filename)
        elif content_type.startswith("audio/"):
            part = MIMEAudio(data, name=filename)
        else:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(data)
            email.encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(part)


def build_services(config: dict) -> dict:
    """Build services."""
    return {"gmail": GmailService()}
