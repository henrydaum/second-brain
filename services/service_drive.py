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

**Signing in happens on the poll thread, and that is the whole of why boot no
longer stops.** ``start`` used to end in ``run_local_server``, and services are
auto-loaded on the *boot thread* — so a first run hung the entire app behind a
browser window, with no frontend started and therefore no surface anywhere to
say why. A plugin cannot spawn its own thread to get out of that (``threading``
is refused: the kernel schedules), but it does not need one. The kernel already
owns a thread per resident service for :meth:`poll`, it is started after
``start`` returns, and its first tick is immediate. So ``start`` does only what
it can do without a human — read a stored token, refresh an expired one — and
the browser opens on the first poll instead, next to a boot that has carried on
without it.

The attempt **announces itself before it blocks**, because a browser window
appearing unbidden is only obvious to somebody already looking at the screen.
At boot the notification usually goes nowhere — the frontends are still
starting, and there is nothing subscribed to hear it — and that is accepted
rather than worked around. The moment that actually matters is *installing*
this package: the frontends are up, the user just typed ``/packages install``,
and the notification lands where they are already looking.

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
    # The OAuth dance waits on a human in a browser, and that wait is *guest*
    # time — not time blocked on the kernel — so it is charged in full against
    # this deadline. 600 is the most it can usefully be: the interpreter clamps
    # a declared timeout to MAX_TIMEOUT_SECONDS and the watchdog's HARD_CEILING
    # ends any call at ten minutes of wall clock regardless. So ten minutes is
    # how long the sign-in window stays open, and asking for more would only
    # look like it worked.
    timeout = 600
    requests = ["paths.get", "fs.read", "fs.write", "fs.list", "session.push"]
    exports = ["download_as", "download_text", "download_csv", "describe"]

    # Long, because after the one sign-in attempt every tick is a no-op, and a
    # no-op is still a round trip into the box. The first tick is immediate —
    # the poll loop calls before it ever waits — which is the only timing this
    # service actually depends on.
    poll_interval = 3600.0

    def __init__(self):
        """Nothing is acquired until start()."""
        self._creds = None
        self._attempted = False

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Load, using a stored token when there is a usable one.

        Deliberately does **not** open a browser. Everything here is either
        local or one HTTPS call against a credential we already hold, so this
        returns in milliseconds and the boot thread moves on; the sign-in that
        needs a person is :meth:`poll`'s.

        Having no token is not a failure — it returns True with ``_creds``
        unset, which is the honest state: *installed, not yet signed in*.
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
                # an error worth failing on — the first poll replaces it.
                sdk.log(f"stored token unusable, signing in on the first poll:"
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
            # No human and no browser — one HTTPS call against a token we
            # already hold — so it stays at load, where it keeps the ordinary
            # restart silent.
            from google.auth.exceptions import GoogleAuthError

            try:
                creds.refresh(Request())
            except GoogleAuthError as exc:
                # The expected end of every refresh token, and the reason this
                # is caught rather than left to raise: Google expires them
                # after a week while the OAuth app sits in "Testing", so this
                # is the *ordinary* weekly path, not a fault. Raising here
                # failed the load outright — and a service that will not load
                # never reaches the poll that could have signed it back in, so
                # the one recoverable failure was the one that locked the door.
                sdk.log(f"stored token could not be refreshed, signing in on "
                        f"the first poll: {exc}", level="warning")
            else:
                sdk.fs.write(token_path, creds.to_json())
                self._creds = creds
                sdk.log("google drive token refreshed")
                return True

        sdk.log("google drive is installed but not signed in; the first poll "
                "will open a browser")
        return True

    def poll(self, sdk):
        """Sign in, once per load, off the boot thread.

        This is the blocking half, and it is here rather than in ``start``
        purely for *which thread it is on*: the kernel drives poll on a thread
        of its own that it starts after ``start`` returns, so the app finishes
        booting and the frontends come up while the browser waits. The block
        itself is unavoidable — ``run_local_server`` binds a port and waits for
        a redirect, which is the OAuth library's design, not ours.

        **One attempt per load**, because a retry loop around something that
        opens a browser is a browser that opens over and over. If the window is
        missed, the way back is to reload the service (``/services`` → Load),
        which is a new box and therefore a fresh attempt — and the failure
        notification says so.

        Returns falsy always: there is never work to drain, so the loop sleeps
        ``poll_interval`` between the no-ops that follow the one real tick.
        """
        if self._attempted or (self._creds and self._creds.valid):
            return False
        self._attempted = True
        try:
            self._authenticate(sdk)
        except Exception as exc:
            # Swallowed rather than raised, and this is the one place in this
            # file where that is right: a raising poll counts against
            # max_poll_failures, and five failures stop the loop for the life
            # of the process — so an unreachable Google would end the poll
            # thread over something that will be true again in a minute. The
            # user is told, which is what the raise would have been for.
            sdk.log(f"google drive sign-in failed: {exc}", level="error")
            self._notify(
                sdk,
                "Google Drive could not be signed in",
                f"The sign-in did not complete: {exc}\n\n"
                "Drive is installed but cannot answer until it is signed in. "
                "To try again, reload the service from `/services` — pick "
                "**google_drive**, then **Load it** — while you are at this "
                "computer.",
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
        """
        from google_auth_oauthlib.flow import InstalledAppFlow

        secret_path = self._secret_path(sdk)
        if not _exists(sdk, secret_path):
            raise RuntimeError(
                f"no {CLIENT_SECRET_FILE} at {secret_path}: download the OAuth "
                f"client secret from the Google Cloud Console and save it "
                f"there")

        self._notify(
            sdk,
            "Google Drive needs authorizing",
            "A browser window is opening so you can sign in to Google Drive.\n\n"
            "The sign-in has to be completed **on this computer** — the one "
            "Second Brain is running on — because Google hands the token back "
            "to a local port here. Signing in on a phone or another machine "
            "will look like it worked and will not connect.\n\n"
            "The window stays open for ten minutes. This happens once: the "
            "token is saved afterwards and refreshed automatically.",
            level="warning")

        sdk.log("opening a browser to authenticate with Google Drive")
        flow = InstalledAppFlow.from_client_config(
            json.loads(sdk.fs.read(secret_path)), SCOPES)
        creds = flow.run_local_server(port=0)

        sdk.fs.write(self._token_path(sdk), creds.to_json())
        self._creds = creds
        sdk.log("google drive authenticated")
        self._notify(sdk, "Google Drive is connected",
                     "Signed in. Docs and Sheets shortcuts will resolve from "
                     "now on.", level="success")

    def _notify(self, sdk, title: str, body: str, *, level: str) -> None:
        """Raise one notification, and never fail for want of somewhere to put it.

        Guarded because of *when* this runs. On an ordinary boot the poll
        thread starts before the runtime exists, so there is nothing to notify
        through and the Request fails — which must not be the reason a sign-in
        does not happen. It is the same argument ``runtime.notifications.notify``
        makes for its own defensiveness one layer down: a notification must
        never break the thing that had something to say.
        """
        try:
            sdk.session.push(body, title=title, notify=True, level=level)
        except Exception as exc:
            sdk.log(f"could not notify ({title}): {exc}", level="debug")

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

        ``loaded`` and ``authenticated`` come apart now that the sign-in is not
        part of loading: the service loads without credentials and acquires
        them on the first poll, so "installed but not signed in" is a real
        state and worth being able to name.
        """
        return {"loaded": True, "authenticated": self._creds is not None,
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
        # Deliberately does not sign in. A call that opened a browser is the
        # thing this design exists to stop: it puts the window at whatever
        # moment some parser happened to meet a .gdoc, which is exactly the
        # moment nobody is watching for it. Failing here is honest and cheap,
        # and the message says where the window can be asked for on purpose.
        if self._creds is None:
            raise RuntimeError(
                "google drive is not signed in — reload the service from "
                "/services (pick google_drive, then 'Load it') while you are "
                "at this computer, and a browser will open")

        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        if self._creds.expired and self._creds.refresh_token:
            sdk.log("refreshing an expired Drive token", level="debug")
            self._creds.refresh(Request())

        return build("drive", "v3", credentials=self._creds,
                     cache_discovery=False)
