"""
Google Drive Service.

Handles OAuth 2.0 authentication and provides the Google Drive API
service object. Follows the standard service interface (load/unload/loaded)
so it integrates with the services dict.

Usage:
    drive = GoogleDriveService()
    services["drive"] = drive

    # Load when ready (opens browser for OAuth if needed)
    drive.load()

    text = drive.download_text(doc_id)

    # Unload to release
    drive.unload()

Credentials:
    - credentials.json: OAuth client secret (from Google Cloud Console)
      Stored in DATA_DIR (immutable, user provides once)
    - token.json: OAuth refresh token (auto-generated after first login)
      Stored in DATA_DIR (mutable, auto-refreshed)

Store the authenticated credentials (lightweight, thread-safe).
Hand out a fresh build() client per call via get_client().
build() is cheap (~1ms) — the OAuth dance only happens in load().
"""

import os
import time
from pathlib import Path
import logging

from plugins.BaseService import BaseService
from paths import DATA_DIR

logger = logging.getLogger("driveService")

class GoogleDriveService(BaseService):
    def __init__(self):
        super().__init__()
        self.model_name = "google_drive"
        self.shared = False  # Each client is a separate instance (build() is cheap)
        self._creds = None       # Authenticated credentials (thread-safe, reusable)

    @staticmethod
    def _is_connected() -> bool:
        """Check for internet connectivity."""
        try:
            import requests
            requests.head("https://www.google.com", timeout=3)
            return True
        except Exception as e:
            logger.debug(f"Connectivity check failed: {e}")
            return False

    def _load(self) -> bool:
        """
        Authenticate with Google Drive and store credentials.
        Opens a browser for OAuth consent if no valid token exists.
        Returns True if successful.
        """
        if not self._is_connected():
            logger.warning("[Drive] No internet — cannot authenticate.")
            return False

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            logger.error("[Drive] Google client libraries not installed. "
                         "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return False

        cred_path = DATA_DIR / "credentials.json"
        if not cred_path.exists():
            logger.error(f"[Drive] No credentials.json found at {cred_path}")
            return False

        SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
        token_path = DATA_DIR / "token.json"
        creds = None

        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception as e:
                logger.debug(f"Token load failed, will re-auth: {e}")

        try:
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    logger.info("[Drive] Opening browser for authentication...")
                    flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
                    creds = flow.run_local_server(port=0)

                with open(token_path, "w") as f:
                    f.write(creds.to_json())

            # Store creds — the important part
            self._creds = creds

            self.loaded = True
            return True

        except Exception as e:
            logger.error(f"[Drive] Authentication failed: {e}")
            return False

    def get_client(self):
        """
        Return a fresh Drive API client for the caller's exclusive use.

        Each call to build() creates an independent httplib2 transport,
        so concurrent threads won't block each other. The credentials
        object is thread-safe and shared — only the HTTP client is new.

        Returns None if the service isn't loaded.
        """
        if not self.loaded or not self._creds:
            logger.error("[Drive] Not loaded — call load() first.")
            return None

        try:
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            # Refresh token if expired (thread-safe operation on creds)
            if self._creds.expired and self._creds.refresh_token:
                logger.info("[Drive] Refreshing expired token...")
                self._creds.refresh(Request())

            return build("drive", "v3", credentials=self._creds, cache_discovery=False)

        except Exception as e:
            logger.error(f"[Drive] Failed to create client: {e}")
            return None

    def unload(self):
        """Release the Drive API service and credentials."""
        self._creds = None
        self.loaded = False
        logger.info("[Drive] Service unloaded.")
    
    def download_as(self, doc_id: str, mime_type: str) -> bytes | None:
        """
        Download a Google Drive file exported as the given MIME type.

        Uses get_client() internally for thread safety — each call gets
        its own HTTP transport.

        Args:
            doc_id:    Google Drive file ID.
            mime_type: Export MIME type (e.g. "text/plain", "text/csv",
                    "application/pdf").

        Returns raw bytes, or None on failure.
        """
        client = self.get_client()
        if client is None:
            return None

        try:
            import io
            from googleapiclient.http import MediaIoBaseDownload

            logger.debug(f"[Drive] Downloading {doc_id} as {mime_type}...")
            t0 = time.time()
            request = client.files().export_media(fileId=doc_id, mimeType=mime_type)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            buffer.seek(0)
            data = buffer.read()
            logger.info(
                f"[Drive] Downloaded {len(data)} bytes for {doc_id} "
                f"as {mime_type} in {time.time() - t0:.2f}s"
            )
            return data

        except Exception as e:
            logger.error(f"[Drive] Download failed for {doc_id}: {e}")
            return None

    def download_text(self, doc_id: str) -> str | None:
        """Download a Google Doc as plain text. Returns string or None."""
        data = self.download_as(doc_id, "text/plain")
        if data is None:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"[Drive] UTF-8 decode failed for {doc_id}: {e}")
            return None

    def download_csv(self, doc_id: str) -> str | None:
        """Download a Google Sheet as CSV. Returns string or None."""
        data = self.download_as(doc_id, "text/csv")
        if data is None:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"[Drive] UTF-8 decode failed for {doc_id}: {e}")
            return None


def build_services(config: dict) -> dict:
    return {"google_drive": GoogleDriveService()}
