"""Persistent Gmail API service.

OAuth credentials, refresh state, and Google client objects stay inside this
service's isolated subprocess. Callers reach only the declared exports and
receive JSON-shaped values. User-provided ``credentials.json`` and the saved
Google token established by ``service_drive`` lives at
``workspace/drive/token.json``. Gmail reuses it when it already carries the
needed scope, and upgrades it through OAuth when it does not. The legacy
``gmail_token.json`` is read only as a one-time migration source.
"""

dependencies_files = []
dependencies_pip = [
    "google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2",
]

import base64
import json
import mimetypes
import re
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
import email.encoders

from guest.bases import BaseService

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]


def _exists(sdk, path):
    try:
        return bool(sdk.fs.list(path))
    except sdk.Failed:
        return False


class GmailService(BaseService):
    """Authenticate once and expose bounded Gmail operations."""

    name = "gmail"
    description = "Read, send, and label mail through Gmail OAuth."
    # Box calls are serialized, so one credential/client instance is safer
    # and avoids repeating the OAuth flow for different callers.
    shared = True
    timeout = 300
    requests = ["paths.get", "fs.read", "fs.write", "fs.read_bytes", "fs.list"]
    exports = [
        "list_labels", "modify_labels", "get_self_address", "fetch_inbox",
        "search", "get_message", "mark_read", "mark_unread", "send_message",
        "reply_to",
    ]

    def __init__(self):
        self.creds = None
        self.client = None
        self.self_address = ""
        self.labels_cache = None
        self.token_path = ""

    def start(self, sdk):
        """Load or create OAuth credentials and build the Gmail client."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as error:
            raise RuntimeError(
                "Missing Gmail libraries. Reinstall the Gmail bundle: " + str(error))

        data = sdk.paths.get("data")
        credentials_path = sdk.path.join(data, "credentials.json")
        self.token_path = sdk.path.join(
            sdk.paths.get("workspace"), "drive", "token.json")
        legacy_gmail_token = sdk.path.join(data, "gmail_token.json")
        try:
            client_config = json.loads(sdk.fs.read(credentials_path))
        except (sdk.Failed, TypeError, ValueError) as error:
            sdk.log(f"Gmail credentials unavailable at {credentials_path}: {error}",
                    level="error")
            return False

        creds = None
        stored = (self.token_path if _exists(sdk, self.token_path)
                  else legacy_gmail_token if _exists(sdk, legacy_gmail_token)
                  else None)
        token = {}
        try:
            token = json.loads(sdk.fs.read(stored)) if stored else {}
            creds = Credentials.from_authorized_user_info(token, SCOPES)
        except (sdk.Failed, TypeError, ValueError):
            pass

        try:
            recorded_scopes = token.get("scopes") or []
            if isinstance(recorded_scopes, str):
                recorded_scopes = recorded_scopes.split()
            has_scopes = bool(
                creds and all(scope in recorded_scopes for scope in SCOPES))
            if not creds or not creds.valid or not has_scopes:
                if creds and creds.expired and creds.refresh_token:
                    sdk.log("refreshing Gmail OAuth token")
                    creds.refresh(Request())
                if not creds.valid or not creds.has_scopes(SCOPES):
                    sdk.log("opening browser for Gmail OAuth consent")
                    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                    creds = flow.run_local_server(port=0)
                sdk.fs.write(self.token_path, creds.to_json())
            elif stored != self.token_path:
                sdk.fs.write(self.token_path, creds.to_json())
            self.creds = creds
            self.client = build(
                "gmail", "v1", credentials=creds, cache_discovery=False)
            return True
        except Exception as error:
            sdk.log(f"Gmail authentication failed: {error}", level="error")
            self.creds = None
            self.client = None
            return False

    def stop(self, sdk):
        self.creds = None
        self.client = None
        self.self_address = ""
        self.labels_cache = None

    def _ready(self, sdk):
        if not self.creds or not self.client:
            return False
        try:
            from google.auth.transport.requests import Request
            if self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
                sdk.fs.write(self.token_path, self.creds.to_json())
            return True
        except Exception as error:
            sdk.log(f"Gmail token refresh failed: {error}", level="error")
            return False

    def list_labels(self, sdk, force_refresh=False):
        if self.labels_cache is not None and not force_refresh:
            return self.labels_cache
        if not self._ready(sdk):
            return []
        try:
            response = self.client.users().labels().list(userId="me").execute()
            self.labels_cache = [
                {"id": label.get("id", ""), "name": label.get("name", ""),
                 "type": label.get("type", "user")}
                for label in response.get("labels", [])
            ]
            return self.labels_cache
        except Exception as error:
            sdk.log(f"Gmail list_labels failed: {error}", level="error")
            return []

    def modify_labels(self, sdk, message_id, add_ids=None, remove_ids=None):
        return self._modify_labels(sdk, message_id, add_ids or [], remove_ids or [])

    def get_self_address(self, sdk):
        if self.self_address:
            return self.self_address
        if not self._ready(sdk):
            return ""
        try:
            profile = self.client.users().getProfile(userId="me").execute()
            self.self_address = str(profile.get("emailAddress") or "").strip()
            return self.self_address
        except Exception as error:
            sdk.log(f"Gmail getProfile failed: {error}", level="error")
            return ""

    def fetch_inbox(self, sdk, max_results=50, label="INBOX"):
        if not self._ready(sdk):
            return []
        try:
            found = self.client.users().messages().list(
                userId="me", labelIds=[label],
                maxResults=_limit(max_results)).execute().get("messages", [])
            return [self._summary(self._get(message["id"], "metadata"))
                    for message in found]
        except Exception as error:
            sdk.log(f"Gmail fetch_inbox failed: {error}", level="error")
            return []

    def search(self, sdk, query, max_results=50):
        if not self._ready(sdk):
            return []
        try:
            found = self.client.users().messages().list(
                userId="me", q=str(query or ""),
                maxResults=_limit(max_results)).execute().get("messages", [])
            return [self._summary(self._get(message["id"], "metadata"))
                    for message in found]
        except Exception as error:
            sdk.log(f"Gmail search failed: {error}", level="error")
            return []

    def get_message(self, sdk, message_id):
        if not self._ready(sdk):
            return None
        try:
            return self._parse(self._get(message_id, "full"))
        except Exception as error:
            sdk.log(f"Gmail get_message failed for {message_id}: {error}",
                    level="error")
            return None

    def mark_read(self, sdk, message_id):
        return self._modify_labels(sdk, message_id, [], ["UNREAD"])

    def mark_unread(self, sdk, message_id):
        return self._modify_labels(sdk, message_id, ["UNREAD"], [])

    def send_message(self, sdk, to, subject, body, cc="", attachments=None,
                     from_address=None):
        if not self._ready(sdk):
            return None
        try:
            message = _message(to, subject, body, cc, from_address)
            _attach_files(sdk, message, attachments or [])
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            sent = self.client.users().messages().send(
                userId="me", body={"raw": raw}).execute()
            sdk.log(f"sent Gmail message {sent.get('id')} to {to}")
            return sent.get("id")
        except Exception as error:
            sdk.log(f"Gmail send_message failed: {error}", level="error")
            return None

    def reply_to(self, sdk, message_id, body, attachments=None,
                 from_address=None):
        original = self.get_message(sdk, message_id)
        if not original or not self._ready(sdk):
            return None
        try:
            recipient = _address(original.get("sender", ""))
            subject = original.get("subject", "")
            if not subject.lower().startswith(("re: ", "fwd: ")):
                subject = "Re: " + subject
            message = _message(recipient, subject, body, "", from_address)
            header = original.get("message_id_header") or f"<{message_id}>"
            message["In-Reply-To"] = header
            message["References"] = (
                f"{original.get('references', '')} {header}".strip())
            _attach_files(sdk, message, attachments or [])
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            sent = self.client.users().messages().send(
                userId="me", body={"raw": raw,
                                   "threadId": original.get("thread_id")}).execute()
            return sent.get("id")
        except Exception as error:
            sdk.log(f"Gmail reply_to failed: {error}", level="error")
            return None

    def _get(self, message_id, format_name):
        return self.client.users().messages().get(
            userId="me", id=message_id, format=format_name).execute()

    def _modify_labels(self, sdk, message_id, add, remove):
        if not self._ready(sdk):
            return False
        try:
            self.client.users().messages().modify(
                userId="me", id=message_id,
                body={"addLabelIds": list(add),
                      "removeLabelIds": list(remove)}).execute()
            return True
        except Exception as error:
            sdk.log(f"Gmail label update failed for {message_id}: {error}",
                    level="error")
            return False

    @staticmethod
    def _summary(message):
        headers = _headers(message)
        return {
            "message_id": message.get("id", ""),
            "thread_id": message.get("threadId", ""),
            "subject": headers.get("Subject", ""),
            "sender": headers.get("From", ""),
            "received_at": int(message.get("internalDate", 0)) / 1000.0,
            "snippet": message.get("snippet", ""),
            "is_read": "UNREAD" not in message.get("labelIds", []),
            "labels": message.get("labelIds", []),
        }

    @staticmethod
    def _parse(message):
        headers = _headers(message)
        plain, html = _body_parts(message.get("payload") or {})
        return {
            **GmailService._summary(message),
            "recipients": headers.get("To", ""), "cc": headers.get("Cc", ""),
            "body_plain": plain, "body_html": html,
            "message_id_header": headers.get("Message-ID", ""),
            "references": headers.get("References", ""),
        }


def _limit(value):
    try:
        return max(1, min(int(value), 100))
    except (TypeError, ValueError):
        return 20


def _headers(message):
    return {item.get("name", ""): item.get("value", "")
            for item in (message.get("payload") or {}).get("headers", [])}


def _body_parts(part):
    plain = html = ""
    data = (part.get("body") or {}).get("data")
    if data:
        decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        plain = decoded if part.get("mimeType") == "text/plain" else ""
        html = decoded if part.get("mimeType") == "text/html" else ""
    for child in part.get("parts", []):
        child_plain, child_html = _body_parts(child)
        plain, html = plain or child_plain, html or child_html
    return plain, html


def _message(to, subject, body, cc, from_address):
    message = MIMEMultipart()
    message["To"] = str(to)
    message["Subject"] = str(subject)
    message["From"] = from_address or "me"
    message["Date"] = formatdate(localtime=True)
    if cc:
        message["Cc"] = str(cc)
    message.attach(MIMEText(str(body), "plain"))
    return message


def _attach_files(sdk, message, attachments):
    for path in attachments:
        data = sdk.fs.read_bytes(path)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        main_type, sub_type = content_type.split("/", 1)
        part = MIMEBase(main_type, sub_type)
        part.set_payload(data)
        email.encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment",
                        filename=sdk.path.name(path))
        message.attach(part)


def _address(header):
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", header or "")
    return match.group(0) if match else str(header or "")
