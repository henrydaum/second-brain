"""Persistent OAuth token owner for the ChatGPT Codex backend."""

dependencies_files = []
dependencies_pip = []

import base64
import json
from datetime import datetime, timezone

from guest.bases import BaseService


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REFRESH_SKEW_SECONDS = 120


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _jwt_claims(token):
    try:
        part = token.split(".")[1]
        raw = base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _quote(value):
    out = []
    for byte in str(value).encode("utf-8"):
        char = chr(byte)
        out.append(char if char.isalnum() or char in "-._~" else f"%{byte:02X}")
    return "".join(out)


def _form(values):
    return "&".join(f"{_quote(key)}={_quote(value)}" for key, value in values.items())


class CodexAuthService(BaseService):
    """Keep a private Codex OAuth session fresh for the LLM backend."""

    name = "codex_auth"
    description = "Refresh and supply ChatGPT Codex OAuth credentials."
    shared = True
    poll_interval = 60.0
    max_poll_failures = 100
    exports = ["access_token", "status", "reload", "refresh", "logout"]
    requests = ["config.read", "config.write", "secret.reveal", "net.http"]
    config_settings = [
        (
            "Codex OAuth state",
            "secret_codex_oauth_state",
            "Device-code credentials used by the Codex LLM backend.",
            "",
            {"type": "text", "hidden": True},
        ),
    ]

    def __init__(self):
        self._state = {}
        self._last_error = ""

    def start(self, sdk):
        self._load(sdk)
        if self._state and self._expiring():
            try:
                self._refresh(sdk)
            except Exception as exc:
                self._last_error = str(exc)
                sdk.log(f"Codex token refresh at startup failed: {exc}", level="warning")
        return True

    def stop(self, sdk):
        self._state = {}
        return None

    def poll(self, sdk):
        if self._state and self._expiring():
            try:
                self._refresh(sdk)
            except Exception as exc:
                self._last_error = str(exc)
                sdk.log(f"Codex token refresh failed: {exc}", level="warning")
        return False

    def reload(self, sdk):
        self._load(sdk)
        return self.status(sdk)

    def access_token(self, sdk):
        if not self._state:
            self._load(sdk)
        if not self._state:
            raise RuntimeError("Codex is not signed in. Run /codex and choose Sign in.")
        if self._expiring():
            self._refresh(sdk)
        token = self._state.get("access_token") or ""
        if not token:
            raise RuntimeError("Codex credentials contain no access token. Run /codex and sign in again.")
        return token

    def refresh(self, sdk):
        if not self._state:
            self._load(sdk)
        if not self._state:
            raise RuntimeError("Codex is not signed in. Run /codex and choose Sign in.")
        self._refresh(sdk)
        return self.status(sdk)

    def logout(self, sdk):
        self._state = {}
        self._last_error = ""
        sdk.config.write("secret_codex_oauth_state", "", scope="plugin")
        return True

    def status(self, sdk):
        if not self._state:
            return {"signed_in": False, "last_error": self._last_error}
        claims = _jwt_claims(self._state.get("access_token") or "")
        auth = claims.get("https://api.openai.com/auth") or {}
        return {
            "signed_in": bool(self._state.get("access_token")),
            "account_id": auth.get("chatgpt_account_id") or "",
            "expires_at": self._expires_at(),
            "last_refresh": self._state.get("last_refresh") or "",
            "last_error": self._last_error,
        }

    def _load(self, sdk):
        raw = sdk.secrets.reveal("secret_codex_oauth_state")
        if not raw:
            self._state = {}
            return
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Stored Codex credentials are invalid: {exc}")
        if not isinstance(value, dict):
            raise RuntimeError("Stored Codex credentials are not an object.")
        self._state = value

    def _expires_at(self):
        claims = _jwt_claims(self._state.get("access_token") or "")
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return float(exp)
        value = self._state.get("expires_at")
        return float(value) if isinstance(value, (int, float)) else 0.0

    def _expiring(self):
        expires = self._expires_at()
        return not expires or expires <= _now() + REFRESH_SKEW_SECONDS

    def _refresh(self, sdk):
        refresh_token = self._state.get("refresh_token") or ""
        if not refresh_token:
            raise RuntimeError("Codex refresh token is missing. Run /codex and sign in again.")
        try:
            response = sdk.net.http_json(
                TOKEN_URL,
                method="POST",
                body=_form({
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CLIENT_ID,
                }),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "SecondBrain/1",
                },
            )
        except Exception as exc:
            raise RuntimeError(f"Codex token refresh could not reach OpenAI: {exc}")
        status = response.get("status", 0)
        if status == 429:
            raise RuntimeError("OpenAI rate-limited the Codex token refresh; credentials remain stored.")
        if status != 200:
            message = ""
            try:
                body = response.get("body") or {}
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict):
                    message = error.get("message") or error.get("code") or ""
                elif isinstance(error, str):
                    message = body.get("error_description") or error
            except Exception:
                pass
            suffix = f": {message}" if message else ""
            raise RuntimeError(
                f"Codex token refresh failed with HTTP {status}{suffix}. "
                "Run /codex and sign in again if this persists."
            )
        payload = response.get("body") or {}
        access_token = payload.get("access_token") if isinstance(payload, dict) else ""
        if not access_token:
            raise RuntimeError("Codex token refresh returned no access token.")
        updated = dict(self._state)
        updated["access_token"] = access_token
        updated["refresh_token"] = payload.get("refresh_token") or refresh_token
        updated["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            updated["expires_at"] = _now() + float(expires_in)
        self._state = updated
        self._last_error = ""
        sdk.config.write(
            "secret_codex_oauth_state", json.dumps(updated), scope="plugin"
        )
