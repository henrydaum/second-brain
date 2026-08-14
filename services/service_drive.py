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

**Loading and signing in are separate, because one of them waits on a human.**
``start`` used to end in ``run_local_server``, and extension services are
auto-loaded on the boot thread *before* any frontend starts — so a first run
hung the entire app behind a browser window, with no surface anywhere to say
why. ``start`` now does only what it can do alone (read a stored token, refresh
an expired one) and returns True having signed in or not; ``_ensure_auth``
does the rest at the first call that needs credentials, where there is somebody
to notify and something to notify them on. It is still blocking — that part is
the OAuth library's, not ours — but it is once, and it announces itself.

The connectivity probe is gone — it opened a socket to google.com to decide
whether authenticating was worth attempting, and a machine with DNS but no
route to Google passed it and failed anyway. Attempting the thing and reporting
the failure is both simpler and more accurate. Same call as ``service_embed``.

Credentials:
    - ``credentials.json`` at the root of DATA_DIR — the OAuth client secret
      from Google Cloud Console. You provide this once; nothing here writes it,
      and reading it needs no approval.
    - ``token.json`` under ``workspace/drive/`` — the refresh token, written
      after the first login and refreshed in place.

The token lives in the workspace tree rather than beside the client secret
because everything under DATA_DIR *except* the workspace is protected by
policy, and a service loads unattended: an unattended chain is refused rather
than asked, so writing to the DATA_DIR root was not a dialog nobody answered,
it was a hard denial and the service could never finish starting. The workspace
is freely writable, so the write is SAFE at boot and after every refresh. A
token left at the old path is still read once, then rewritten to the new one.
"""


dependencies_files = []
dependencies_pip = ['google-api-python-client', 'google-auth-oauthlib', 'requests']

import json

from guest.bases import BaseService

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

CLIENT_SECRET_FILE = "credentials.json"
TOKEN_FILE = "token.json"
TOKEN_DIR = "drive"


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
    requests = ["paths.get", "fs.read", "fs.write", "fs.list", "session.push"]
    exports = ["download_as", "download_text", "download_csv", "describe"]

    def __init__(self):
        """Nothing is acquired until start()."""
        self._creds = None

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Load, using a stored token when there is a usable one.

        Deliberately does **not** log in. Loading and authenticating used to be
        one act, and the act blocked: ``run_local_server`` binds a port, opens
        a browser and waits for a human. Extension services are auto-loaded on
        the boot thread, *before* the frontends start — so a first run stopped
        the whole app with a browser window and nothing on screen to explain
        it, and there was no surface anywhere to say anything on.

        So the three settled cases stay here, and only the one that needs a
        person moves to :meth:`_ensure_auth`. Having no token is not a failure:
        it returns True with ``_creds`` unset, which is the honest state —
        *installed, not yet signed in* — and the sign-in happens at the first
        call that needs it, by which time somebody is watching.

        Returns False rather than raising for every *expected* absence, because
        a service that cannot start is a capability the user has not set up
        yet, not a fault. An unexpected failure is left to raise, where it
        arrives with a traceback.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        secret_path = self._secret_path(sdk)
        token_path = self._token_path(sdk)
        legacy_token = sdk.path.join(sdk.paths.get("data"), TOKEN_FILE)

        if not _exists(sdk, secret_path):
            sdk.log(f"no {CLIENT_SECRET_FILE} at {secret_path}; get one from "
                    "the Google Cloud Console and place it there",
                    level="error")
            return False

        creds = None
        stored = (token_path if _exists(sdk, token_path)
                  else legacy_token if _exists(sdk, legacy_token) else None)
        if stored:
            try:
                creds = Credentials.from_authorized_user_info(
                    json.loads(sdk.fs.read(stored)), SCOPES)
            except (ValueError, KeyError) as exc:
                # A token written by an older scope set, or half-written. Not
                # an error worth failing on — _ensure_auth replaces it.
                sdk.log(f"stored token unusable, re-authenticating on first use:"
                        f" {exc}", level="debug")

        if creds and creds.valid:
            if stored != token_path:
                # Valid, but at the old address. Copy it across now rather than
                # waiting for expiry, so the move happens once instead of
                # falling back through this branch on every boot.
                sdk.fs.write(token_path, creds.to_json())
            self._creds = creds
            return True

        if creds and creds.expired and creds.refresh_token:
            # Refreshing needs no human and no browser — it is one HTTPS call
            # against a token we already hold — so it stays at load, where it
            # keeps the ordinary restart silent.
            creds.refresh(Request())
            sdk.fs.write(token_path, creds.to_json())
            self._creds = creds
            sdk.log("google drive token refreshed")
            return True

        sdk.log("google drive is installed but not signed in; will ask on "
                "first use")
        return True

    def _ensure_auth(self, sdk):
        """Sign in if we are not already, telling the user before we block.

        The blocking half of the old ``start``. It is still blocking — the OAuth
        library binds a local port and waits for a browser redirect, which is
        foreign I/O past the kernel's reach — but it now happens somewhere a
        person can be told about it, and it happens once.

        The notification says *where* to sign in, because that is the part
        nobody can guess and the part that silently fails: the token comes back
        to a port on this machine, so signing in on a phone, or over SSH from a
        laptop, hands the credential to a listener that is not us.
        """
        if self._creds and self._creds.valid:
            return

        from google_auth_oauthlib.flow import InstalledAppFlow

        secret_path = self._secret_path(sdk)
        if not _exists(sdk, secret_path):
            raise RuntimeError(
                f"google drive has no {CLIENT_SECRET_FILE}: download the OAuth "
                f"client secret from the Google Cloud Console and save it at "
                f"{secret_path}")

        sdk.session.push(
            "A browser window is opening so you can sign in to Google Drive.\n\n"
            "The sign-in has to be completed **on this computer** — the one "
            "Second Brain is running on — because Google hands the token back "
            "to a local port here. Signing in on a phone or another machine "
            "will look like it worked and will not connect.\n\n"
            "This happens once. The token is saved afterwards and refreshed "
            "automatically.",
            title="Google Drive needs authorizing",
            notify=True, level="warning")

        sdk.log("opening a browser to authenticate with Google Drive")
        flow = InstalledAppFlow.from_client_config(
            json.loads(sdk.fs.read(secret_path)), SCOPES)
        creds = flow.run_local_server(port=0)

        sdk.fs.write(self._token_path(sdk), creds.to_json())
        self._creds = creds
        sdk.log("google drive authenticated")

    @staticmethod
    def _secret_path(sdk):
        """Where the OAuth client secret lives. Yours to provide, never written."""
        return sdk.path.join(sdk.paths.get("data"), CLIENT_SECRET_FILE)

    @staticmethod
    def _token_path(sdk):
        """Where the refresh token lives — inside the freely writable workspace.

        Everything under DATA_DIR *except* the workspace is protected by policy,
        and a service acts unattended: an unattended chain is refused rather
        than asked, so the DATA_DIR root was not a dialog nobody answered, it
        was a hard denial.
        """
        return sdk.path.join(sdk.paths.get("workspace"), TOKEN_DIR, TOKEN_FILE)

    def stop(self, sdk):
        """Drop the credentials. The box closing is what releases the rest."""
        self._creds = None
        return None

    # ── exports ─────────────────────────────────────────────────────

    def describe(self, sdk):
        """Whether this service can currently answer.

        ``loaded`` and ``authenticated`` came apart when the sign-in moved out
        of ``start``: the service now loads without credentials and acquires
        them on first use, so "installed but not signed in" is a real state and
        worth being able to name.
        """
        signed_in = self._creds is not None
        return {"loaded": True, "authenticated": signed_in,
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
        self._ensure_auth(sdk)

        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        if self._creds.expired and self._creds.refresh_token:
            sdk.log("refreshing an expired Drive token", level="debug")
            self._creds.refresh(Request())

        return build("drive", "v3", credentials=self._creds,
                     cache_discovery=False)
