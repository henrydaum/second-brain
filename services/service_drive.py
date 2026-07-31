"""Google Drive: OAuth once, then export Docs and Sheets as text.

Two parsers delegate here — ``parse_gdoc`` for ``.gdoc`` shortcuts and
``parse_tabular`` for ``.gsheet`` — so this service is what makes a Drive
shortcut on disk behave like the document it points at.

**What crosses and what does not.** The native version's central object was
``get_client()``, handing each caller a fresh ``googleapiclient`` transport
because ``build()`` is cheap and the credentials are thread-safe. None of that
survives a box: a live API client is exactly the kind of thing that cannot
cross the boundary, so it can never be an export. The client is private now,
built per call for the same reason it always was, and the exports answer with
bytes and strings. The service is ``shared`` for the same reason it was not
before, read the other way round — a box serializes its calls, so per-caller
instances buy nothing, and a second instance would mean a second OAuth dance.

**Where the boundary genuinely stops.** ``run_local_server`` binds a port and
opens a browser; the API client makes its own HTTPS calls. Both are foreign
libraries doing their own I/O, past the kernel's reach and documented as the
limit of what the contract covers. What is *not* given away: the two credential
files are read through ``sdk.fs.read`` and handed to the library's
``from_client_config`` / ``from_authorized_user_info`` constructors rather than
its ``*_file`` variants, so the paths are the kernel's and the reads are
mediated even though what happens next is not.

The connectivity probe is gone — it opened a socket to google.com to decide
whether authenticating was worth attempting, and a machine with DNS but no
route to Google passed it and failed anyway. Attempting the thing and reporting
the failure is both simpler and more accurate. Same call as ``service_embed``.

Credentials, both at the root of DATA_DIR:
    - ``credentials.json`` — the OAuth client secret from Google Cloud Console.
      You provide this once; nothing here writes it.
    - ``token.json`` — the refresh token, written after the first login and
      refreshed in place.
"""


dependencies_files = []
dependencies_pip = ['google-api-python-client', 'google-auth-oauthlib', 'requests']

import json

from guest.bases import BaseService

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

CLIENT_SECRET_FILE = "credentials.json"
TOKEN_FILE = "token.json"


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


class GoogleDriveService(BaseService):
    """Authenticated read-only access to Google Drive."""

    name = "google_drive"
    description = "Export Google Docs and Sheets as text."
    shared = True
    timeout = 300           # the OAuth dance waits on a human in a browser
    requests = ["paths.get", "fs.read", "fs.write", "fs.list"]
    exports = ["download_as", "download_text", "download_csv", "describe"]

    def __init__(self):
        """Nothing is acquired until start()."""
        self._creds = None

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Authenticate, reusing a stored token when one is still good.

        Returns False rather than raising for every *expected* absence — no
        client secret, a token that will not load — because a service that
        cannot start is a capability the user has not set up yet, not a fault.
        An unexpected failure is left to raise, where it arrives with a
        traceback.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        data = sdk.paths.get("data")
        secret_path = sdk.path.join(data, CLIENT_SECRET_FILE)
        token_path = sdk.path.join(data, TOKEN_FILE)

        if not _exists(sdk, secret_path):
            sdk.log(f"no {CLIENT_SECRET_FILE} at {secret_path}; get one from "
                    "the Google Cloud Console and place it there",
                    level="error")
            return False

        creds = None
        if _exists(sdk, token_path):
            try:
                creds = Credentials.from_authorized_user_info(
                    json.loads(sdk.fs.read(token_path)), SCOPES)
            except (ValueError, KeyError) as exc:
                # A token written by an older scope set, or half-written. Not
                # an error worth failing on — the flow below replaces it.
                sdk.log(f"stored token unusable, re-authenticating: {exc}",
                        level="debug")

        if creds and creds.valid:
            self._creds = creds
            return True

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            sdk.log("opening a browser to authenticate with Google Drive")
            flow = InstalledAppFlow.from_client_config(
                json.loads(sdk.fs.read(secret_path)), SCOPES)
            creds = flow.run_local_server(port=0)

        sdk.fs.write(token_path, creds.to_json())
        self._creds = creds
        sdk.log("google drive authenticated")
        return True

    def stop(self, sdk):
        """Drop the credentials. The box closing is what releases the rest."""
        self._creds = None
        return None

    # ── exports ─────────────────────────────────────────────────────

    def describe(self, sdk):
        """Whether this service can currently answer."""
        return {"loaded": self._creds is not None,
                "scopes": list(SCOPES)}

    def download_as(self, sdk, doc_id: str, mime_type: str):
        """Export one Drive file as ``mime_type``, answering with raw bytes.

        Bytes cross the boundary natively, so this is exportable as it stands
        — a caller wanting a PDF of a Doc gets the file, not a description of
        one. The text and CSV helpers below are the two decodes the parsers
        actually want.
        """
        import io

        from googleapiclient.http import MediaIoBaseDownload

        client = self._client(sdk)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(
            buffer, client.files().export_media(fileId=doc_id,
                                                mimeType=mime_type))
        done = False
        while not done:
            _status, done = downloader.next_chunk()

        data = buffer.getvalue()
        sdk.log(f"downloaded {len(data)} bytes for {doc_id} as {mime_type}",
                level="debug")
        return data

    def download_text(self, sdk, doc_id: str):
        """A Google Doc as plain text, or None if it does not decode."""
        return self._decode(sdk, doc_id, "text/plain")

    def download_csv(self, sdk, doc_id: str):
        """A Google Sheet as CSV, or None if it does not decode."""
        return self._decode(sdk, doc_id, "text/csv")

    # ── internals ───────────────────────────────────────────────────

    def _decode(self, sdk, doc_id: str, mime_type: str):
        """Download and decode as UTF-8.

        None for a decode failure only. A download failure raises, because the
        caller can tell those apart and should: one means the export is not
        text, the other means Drive said no.
        """
        data = self.download_as(sdk, doc_id, mime_type)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            sdk.log(f"utf-8 decode failed for {doc_id}: {exc}", level="error")
            return None

    def _client(self, sdk):
        """A Drive API client, refreshing the token if it has expired.

        Private, and it has to be: this returns a live ``googleapiclient``
        object, which is the one thing a service may never hand across the
        boundary. Built per call because ``build()`` is ~1ms and each one
        carries its own transport.
        """
        if self._creds is None:
            raise RuntimeError("google drive is not authenticated")

        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        if self._creds.expired and self._creds.refresh_token:
            sdk.log("refreshing an expired Drive token", level="debug")
            self._creds.refresh(Request())

        return build("drive", "v3", credentials=self._creds,
                     cache_discovery=False)
