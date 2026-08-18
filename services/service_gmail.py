"""Persistent Gmail API service.

OAuth credentials, refresh state, and Google client objects stay inside this
service's isolated subprocess. Callers reach only the declared exports and
receive JSON-shaped values.

Credentials:
    - ``credentials.json`` at the root of DATA_DIR — the OAuth client secret
      you provide once. Nothing here writes it.
    - ``token.json`` under ``workspace/drive/`` — the same token
      ``service_drive`` establishes, shared because it is one Google account
      and one consent screen. Gmail reuses it when it already carries
      ``gmail.modify`` and asks for a fresh consent when it does not. The
      legacy ``gmail_token.json`` beside the client secret is read once as a
      migration source and then rewritten to the new path.

The token lives in the workspace tree rather than beside the client secret
because everything under DATA_DIR *except* the workspace is protected by
policy, and a service loads unattended: an unattended chain is refused rather
than asked, so writing to the DATA_DIR root was a hard denial, not a dialog
nobody answered.

**A scope is only what Google granted.** Reading the token with
``from_authorized_user_info(info, SCOPES)`` *sets* those scopes on the object,
which makes ``creds.has_scopes(SCOPES)`` a tautology and makes
``creds.to_json()`` write a record of a grant that was never made. Both halves
of that failed silently and in opposite directions: this service skipped the
consent upgrade a Drive-only token needed and then overwrote the honest record
with a false one, after which Gmail could 403 forever without ever asking
again; while ``service_drive``, reading the same file with only its own scope,
downgraded the record of a token that legitimately carried both and forced a
re-consent on every refresh. The file's own ``scopes`` key is the only account
of what was consented to, so it is read as written and never widened.

**Signing in is poll's, not start's.** Extension services auto-load on the boot
thread, so ``run_local_server`` here stopped the whole app behind a browser
window with nothing on screen to explain it. ``start`` now does only what it
can do without a human — read a stored token, refresh an expired one — and
returns in milliseconds; the kernel starts the poll thread afterwards and its
first tick is immediate. This is the split ``service_drive`` settled on in
c58f1ea7, and the reasoning there applies here unchanged.
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

CLIENT_SECRET_FILE = "credentials.json"
TOKEN_FILE = "token.json"
TOKEN_DIR = "drive"
LEGACY_TOKEN_FILE = "gmail_token.json"


def _exists(sdk, path) -> bool:
    """Whether a path is there.

    ``sdk.fs.list`` *fails* on a missing path rather than answering with an
    empty list, and the SDK turns a failed Request into a raise — so
    ``if sdk.fs.list(p)`` does not test existence, it throws.
    """
    try:
        return bool(sdk.fs.list(path))
    except sdk.Failed:
        return False


def _stored_credentials(sdk, Credentials, path):
    """Credentials carrying the scopes the token file actually records.

    No ``scopes`` argument, deliberately — see the note in the module
    docstring. A file recording no scopes at all answers False to every
    ``has_scopes`` check, which costs one consent screen and is the safe end:
    the alternative is assuming a grant and discovering it was wrong as a 403
    from inside a tool call.
    """
    try:
        info = json.loads(sdk.fs.read(path))
    except (sdk.Failed, TypeError, ValueError) as error:
        sdk.log(f"stored token at {path} could not be read: {error}",
                level="debug")
        return None
    try:
        return Credentials.from_authorized_user_info(info)
    except (TypeError, ValueError) as error:
        # Written by an older format, or half-written. Not worth failing the
        # load on — the first poll replaces it.
        sdk.log(f"stored token at {path} unusable, signing in on the first "
                f"poll: {error}", level="debug")
        return None


class GmailService(BaseService):
    """Authenticate once and expose bounded Gmail operations."""

    name = "gmail"
    description = "Read, send, and label mail through Gmail OAuth."
    # Box calls are serialized, so one credential/client instance is safer
    # and avoids repeating the OAuth flow for different callers.
    shared = True
    # The OAuth dance waits on a human in a browser, and that wait is *guest*
    # time — not time blocked on the kernel — so it is charged in full against
    # this deadline. 600 is the most it can usefully be: the interpreter clamps
    # a declared timeout to MAX_TIMEOUT_SECONDS and the watchdog's HARD_CEILING
    # ends any call at ten minutes of wall clock regardless. At the 300 this
    # used to declare, the sign-in window closed halfway through.
    timeout = 600
    requests = ["paths.get", "fs.read", "fs.write", "fs.read_bytes", "fs.list",
                "session.push"]
    exports = [
        "list_labels", "modify_labels", "get_self_address", "fetch_inbox",
        "search", "get_message", "mark_read", "mark_unread", "send_message",
        "reply_to", "describe",
    ]

    # Long, because after the one sign-in attempt every tick is a no-op, and a
    # no-op is still a round trip into the box. The first tick is immediate —
    # the poll loop calls before it ever waits — which is the only timing this
    # service actually depends on.
    poll_interval = 3600.0

    def __init__(self):
        self.creds = None
        self.client = None
        self.self_address = ""
        self.labels_cache = None
        self.token_path = ""
        self.credentials_path = ""
        self._attempted = False

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Load, using the stored token when it already carries Gmail access.

        Deliberately does **not** open a browser: everything here is either
        local or one HTTPS call against a credential we already hold, so the
        boot thread moves on and the sign-in that needs a person is
        :meth:`poll`'s.

        Returning True with no client is a real state and the honest one —
        *installed, not yet signed in*. False is kept for the one absence that
        no amount of waiting fixes: a missing client secret, which is yours to
        provide and which nothing here can ask for.
        """
        try:
            from google.oauth2.credentials import Credentials
            # Imported here only to fail early and legibly: a half-installed
            # bundle should say so at load, not as a ModuleNotFoundError out of
            # _adopt once a token has already been read.
            import googleapiclient.discovery  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Missing Gmail libraries. Reinstall the Gmail bundle: "
                + str(error))

        data = sdk.paths.get("data")
        self.credentials_path = sdk.path.join(data, CLIENT_SECRET_FILE)
        self.token_path = sdk.path.join(
            sdk.paths.get("workspace"), TOKEN_DIR, TOKEN_FILE)
        legacy_token = sdk.path.join(data, LEGACY_TOKEN_FILE)

        if not _exists(sdk, self.credentials_path):
            sdk.log(f"no {CLIENT_SECRET_FILE} at {self.credentials_path}; get "
                    f"one from the Google Cloud Console and place it there",
                    level="error")
            return False

        stored = (self.token_path if _exists(sdk, self.token_path)
                  else legacy_token if _exists(sdk, legacy_token) else None)
        creds = _stored_credentials(sdk, Credentials, stored) if stored else None

        if creds and not creds.has_scopes(SCOPES):
            # Drive's own token, or one from an older scope set. A scope cannot
            # be added by refreshing — only a fresh consent grants one — so
            # this is poll's to fix, and the file is left exactly as Drive
            # wrote it meanwhile.
            sdk.log("the stored Google token does not carry Gmail access; the "
                    "first poll will ask for it")
            return True

        if creds and not creds.valid and creds.expired and creds.refresh_token:
            self._refresh(sdk, creds)

        if not creds or not creds.valid:
            sdk.log("gmail is installed but not signed in; the first poll "
                    "will open a browser")
            return True

        if stored != self.token_path:
            # Valid, but at the old address. Copy it across now rather than
            # waiting for expiry, so the move happens once instead of falling
            # through this branch on every boot.
            sdk.fs.write(self.token_path, creds.to_json())
        self._adopt(sdk, creds)
        return True

    def poll(self, sdk):
        """Sign in, once per load, off the boot thread.

        This is the blocking half, and it is here rather than in ``start``
        purely for *which thread it is on*: the kernel drives poll on a thread
        of its own that it starts after ``start`` returns, so the app finishes
        booting and the frontends come up while the browser waits.

        **One attempt per load**, because a retry loop around something that
        opens a browser is a browser that opens over and over. If the window is
        missed, the way back is to reload the service (``/services`` → Load),
        which is a new box and therefore a fresh attempt — and the failure
        notification says so.
        """
        if self._attempted or self.client:
            return False
        self._attempted = True
        try:
            self._authenticate(sdk)
        except Exception as exc:
            # Swallowed rather than raised: a raising poll counts against
            # max_poll_failures, and five failures stop the loop for the life
            # of the process — so an unreachable Google would end the poll
            # thread over something that will be true again in a minute. The
            # user is told, which is what the raise would have been for.
            sdk.log(f"gmail sign-in failed: {exc}", level="error")
            self._notify(
                sdk,
                "Gmail could not be signed in",
                f"The sign-in did not complete: {exc}\n\n"
                "Gmail is installed but cannot read or send until it is "
                "signed in. To try again, reload the service from "
                "`/services` — pick **gmail**, then **Load it** — while you "
                "are at this computer.",
                level="error")
        return False

    def _authenticate(self, sdk):
        """Tell the user what is about to happen, then run the OAuth flow.

        The notification goes first and deliberately: a browser window opening
        by itself is only self-explanatory to somebody watching the screen at
        that second. It says *where* to sign in, because that is the part
        nobody can guess and the part that fails silently — the token comes
        back to a port on this machine, so signing in on a phone, or over SSH
        from a laptop, hands the credential to a listener that is not us.

        The flow asks for Drive's scope alongside Gmail's, so one consent
        serves both services and what this writes is a superset of what
        ``service_drive`` needs.
        """
        from google_auth_oauthlib.flow import InstalledAppFlow

        if not _exists(sdk, self.credentials_path):
            raise RuntimeError(
                f"no {CLIENT_SECRET_FILE} at {self.credentials_path}: download "
                f"the OAuth client secret from the Google Cloud Console and "
                f"save it there")

        self._notify(
            sdk,
            "Gmail needs authorizing",
            "A browser window is opening so you can grant Second Brain access "
            "to Gmail.\n\n"
            "The sign-in has to be completed **on this computer** — the one "
            "Second Brain is running on — because Google hands the token back "
            "to a local port here. Signing in on a phone or another machine "
            "will look like it worked and will not connect.\n\n"
            "The window stays open for ten minutes. This happens once: the "
            "token is saved afterwards and refreshed automatically.",
            level="warning")

        sdk.log("opening a browser to authenticate with Gmail")
        flow = InstalledAppFlow.from_client_config(
            json.loads(sdk.fs.read(self.credentials_path)), SCOPES)
        creds = flow.run_local_server(port=0)

        sdk.fs.write(self.token_path, creds.to_json())
        self._adopt(sdk, creds)
        sdk.log("gmail authenticated")
        self._notify(sdk, "Gmail is connected",
                     "Signed in. Reading, sending and labelling will work "
                     "from now on.", level="success")

    def _refresh(self, sdk, creds) -> bool:
        """One HTTPS call against a credential we already hold."""
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request

        try:
            creds.refresh(Request())
        except GoogleAuthError as error:
            # The expected end of every refresh token, and the reason this is
            # caught rather than left to raise: Google expires them after a
            # week while the OAuth app sits in "Testing", so this is the
            # *ordinary* weekly path, not a fault. Raising here failed the load
            # outright — and a service that will not load never reaches the
            # poll that could have signed it back in, so the one recoverable
            # failure was the one that locked the door.
            sdk.log(f"stored token could not be refreshed, signing in on the "
                    f"first poll: {error}", level="warning")
            return False
        sdk.fs.write(self.token_path, creds.to_json())
        sdk.log("gmail token refreshed")
        return True

    def _adopt(self, sdk, creds) -> bool:
        """Hold the credentials and build the client."""
        from googleapiclient.discovery import build

        try:
            self.client = build("gmail", "v1", credentials=creds,
                                cache_discovery=False)
        except Exception as error:
            sdk.log(f"could not build the Gmail client: {error}", level="error")
            self.creds = self.client = None
            return False
        self.creds = creds
        return True

    def _notify(self, sdk, title: str, body: str, *, level: str) -> None:
        """Raise one notification, and never fail for want of somewhere to put it.

        Guarded because of *when* this runs. On an ordinary boot the poll
        thread starts before the runtime exists, so there is nothing to notify
        through and the Request fails — which must not be the reason a sign-in
        does not happen.
        """
        try:
            sdk.session.push(body, title=title, notify=True, level=level)
        except Exception as exc:
            sdk.log(f"could not notify ({title}): {exc}", level="debug")

    def stop(self, sdk):
        self.creds = None
        self.client = None
        self.self_address = ""
        self.labels_cache = None
        return None

    def describe(self, sdk):
        """Loaded and signed in are different states, so both are reported."""
        return {
            "loaded": True,
            "authenticated": bool(self.creds and self.client),
            "attempted_sign_in": self._attempted,
            "token_path": self.token_path,
        }

    def _ready(self, sdk):
        """Whether a call can be served right now, refreshing if it must."""
        if not self.creds or not self.client:
            return False
        if not self.creds.expired or not self.creds.refresh_token:
            return True
        if self._refresh(sdk, self.creds):
            return True
        sdk.log("gmail token refresh failed; reload the service from "
                "/services to sign in again", level="error")
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
