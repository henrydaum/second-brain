"""
Google Drive Service.

Handles OAuth 2.0 authentication and provides the Google Drive API
service object. Follows the standard service interface (load/unload/loaded)
so it integrates with the services dict and controller.

Usage:
    drive = GoogleDriveService()
    services["drive"] = drive

    # Load when ready (opens browser for OAuth if needed)
    drive.load()

    # Parsers access it via config["_services"]["drive"].service
    api = drive.service
    api.files().list(...).execute()

    # Unload to release
    drive.unload()

Credentials:
    - credentials.json: OAuth client secret (from Google Cloud Console)
      Stored in DATA_DIR (immutable, user provides once)
    - token.json: OAuth refresh token (auto-generated after first login)
      Stored in DATA_DIR (mutable, auto-refreshed)
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("DriveService")

DATA_DIR = Path(os.getenv("LOCALAPPDATA", "")) / "2nd Brain"


class GoogleDriveService:
    def __init__(self):
        self.service = None      # The Google Drive API service object
        self.model_name = "google_drive"
        self.loaded = False

    @staticmethod
    def _is_connected() -> bool:
        """Check for internet connectivity."""
        try:
            import requests
            requests.head("https://www.google.com", timeout=3)
            return True
        except Exception:
            return False

    def load(self) -> bool:
        """
        Authenticate with Google Drive and create the API service.
        Opens a browser for OAuth consent if no valid token exists.
        Returns True if successful.
        """
        # Check internet
        if not self._is_connected():
            logger.warning("[Drive] No internet — cannot authenticate.")
            return False

        # Lazy imports
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            logger.error("[Drive] Google client libraries not installed. "
                         "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return False

        # Check for credentials.json
        cred_path = DATA_DIR / "credentials.json"
        if not cred_path.exists():
            logger.error(f"[Drive] No credentials.json found at {cred_path}")
            return False

        SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
        token_path = DATA_DIR / "token.json"
        creds = None

        # Load existing token
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception:
                pass  # Invalid token, will refresh or re-auth

        # Refresh or re-authenticate
        try:
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    logger.info("[Drive] Opening browser for authentication...")
                    flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
                    creds = flow.run_local_server(port=0)

                # Save token for next time
                with open(token_path, "w") as f:
                    f.write(creds.to_json())

            self.service = build("drive", "v3", credentials=creds, cache_discovery=False)
            self.loaded = True
            logger.info("[Drive] Authenticated successfully.")
            return True

        except Exception as e:
            logger.error(f"[Drive] Authentication failed: {e}")
            return False

    def unload(self):
        """Release the Drive API service."""
        self.service = None
        self.loaded = False
        logger.info("[Drive] Service unloaded.")