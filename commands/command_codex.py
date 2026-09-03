"""Slash command for ChatGPT device-code authentication used by Codex."""

dependencies_files = ['services/service_codex_auth.py']
dependencies_pip = []

import json
import time
from datetime import datetime, timezone

from guest.bases import BaseCommand
from guest.forms import FormStep


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
TOKEN_URL = f"{ISSUER}/oauth/token"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_MODEL = "gpt-5.6-sol"


def _context_size(model):
    name = (model or "").lower()
    if name == "gpt-5.3-codex-spark":
        return 128000
    if name.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.6")):
        return 272000
    return 0


def _quote(value):
    out = []
    for byte in str(value).encode("utf-8"):
        char = chr(byte)
        out.append(char if char.isalnum() or char in "-._~" else f"%{byte:02X}")
    return "".join(out)


def _form(values):
    return "&".join(f"{_quote(key)}={_quote(value)}" for key, value in values.items())


class CodexCommand(BaseCommand):
    """Sign in to Codex with a ChatGPT account and manage that session."""

    name = "codex"
    description = "Sign in to the Codex LLM backend with your ChatGPT plan"
    category = "Capabilities"
    timeout = 600
    approval_actions = ("login", "refresh", "logout")
    approval_actor_id = "user"
    requests = [
        "config.read", "config.write", "service.list", "service.load",
        "service.call", "ui.progress", "net.http",
    ]

    def form(self, sdk, args):
        steps = [FormStep(
            "action",
            "Manage the ChatGPT session used by the Codex LLM backend.",
            True,
            enum=["status", "login", "refresh", "logout"],
            enum_labels=["Show status", "Sign in", "Refresh now", "Sign out"],
        )]
        if args.get("action") == "login":
            steps.append(FormStep(
                "model",
                "Enter the Codex model ID for the LLM profile.",
                True,
                default=DEFAULT_MODEL,
            ))
        return steps

    def run(self, sdk, args):
        action = args.get("action") or "status"
        if action == "status":
            return self._status(sdk)
        if action == "login":
            state = self._device_login(sdk)
            sdk.config.write(
                "secret_codex_oauth_state", json.dumps(state), scope="plugin"
            )
            self._ensure_service(sdk, reload=True)
            model = (args.get("model") or DEFAULT_MODEL).strip()
            self._ensure_profile(sdk, model)
            return (
                "Signed in to ChatGPT for Codex.\n\n"
                f"Created or updated the `{model}` LLM profile. Use `/llm` "
                "to load it or make it the default."
            )
        if action == "refresh":
            self._ensure_service(sdk)
            status = sdk.services.call("codex_auth", "refresh")
            return self._status_text(status, "Codex credentials refreshed.")
        if action == "logout":
            if self._service_loaded(sdk):
                sdk.services.call("codex_auth", "logout")
            else:
                sdk.config.write("secret_codex_oauth_state", "", scope="plugin")
            return "Signed out of ChatGPT for Codex. The LLM profile was kept."
        return f"Unknown Codex action: {action}"

    def _status(self, sdk):
        if not self._service_loaded(sdk):
            if not sdk.config.read("secret_codex_oauth_state", present=True):
                return "Codex is not signed in. Run `/codex` and choose **Sign in**."
            self._ensure_service(sdk)
        return self._status_text(sdk.services.call("codex_auth", "status"))

    def _status_text(self, status, lead=""):
        if not status or not status.get("signed_in"):
            return "Codex is not signed in. Run `/codex` and choose **Sign in**."
        expires = status.get("expires_at") or 0
        if expires:
            expiry = datetime.fromtimestamp(float(expires), tz=timezone.utc).isoformat()
        else:
            expiry = "unknown"
        lines = [lead] if lead else []
        lines.extend(["Codex is signed in.", f"Access token expires: `{expiry}`"])
        if status.get("last_error"):
            lines.append(f"Last refresh error: {status['last_error']}")
        return "\n\n".join(lines)

    def _service_loaded(self, sdk):
        for row in sdk.services.list(details=True) or []:
            if row.get("name") == "codex_auth":
                return bool(row.get("loaded"))
        return False

    def _ensure_service(self, sdk, reload=False):
        if not self._service_loaded(sdk):
            if sdk.services.load("codex_auth") is False:
                raise RuntimeError("The codex_auth service could not be loaded.")
        elif reload:
            sdk.services.call("codex_auth", "reload")

    def _ensure_profile(self, sdk, model):
        profiles = sdk.config.read("llm_profiles") or {}
        profile = dict(profiles.get(model) or {})
        profile.update({
            "llm_endpoint": CODEX_BASE_URL,
            "secret_llm_api_key": "",
            "llm_context_size": _context_size(model),
            "llm_service_class": "CodexBackend",
            "llm_capabilities": {"image": False, "audio": False, "video": False},
        })
        profiles[model] = profile
        sdk.config.write("llm_profiles", profiles, scope="plugin")
        if not sdk.config.read("default_llm_profile"):
            sdk.config.write("default_llm_profile", model, scope="plugin")

    def _device_login(self, sdk):
        headers = {"Accept": "application/json", "User-Agent": "SecondBrain/1"}
        response = sdk.net.http_json(
            f"{ISSUER}/api/accounts/deviceauth/usercode",
            method="POST", headers=headers, json={"client_id": CLIENT_ID})
        if response.get("status") == 429:
            raise RuntimeError(
                "OpenAI rate-limited the device-code request. Wait a minute and try again."
            )
        if response.get("status") != 200:
            raise RuntimeError(
                f"Device-code request failed with HTTP {response.get('status', 0)}."
            )
        data = response.get("body") or {}
        code = data.get("user_code") or ""
        device_id = data.get("device_auth_id") or ""
        if not code or not device_id:
            raise RuntimeError("OpenAI returned an incomplete device code.")
        interval = max(3, int(data.get("interval") or 5))
        sdk.ui.progress(
            "Open https://auth.openai.com/codex/device and enter code "
            f"{code}. Waiting for sign-in…"
        )
        deadline = time.monotonic() + 540
        authorization = None
        while time.monotonic() < deadline:
            time.sleep(interval)
            poll = sdk.net.http_json(
                f"{ISSUER}/api/accounts/deviceauth/token",
                method="POST", headers=headers,
                json={"device_auth_id": device_id, "user_code": code})
            if poll.get("status") == 200:
                authorization = poll.get("body") or {}
                break
            if poll.get("status") not in (403, 404):
                raise RuntimeError(
                    f"Device authorization failed with HTTP {poll.get('status', 0)}."
                )
        if not authorization:
            raise RuntimeError("Device authorization timed out after nine minutes.")
        auth_code = authorization.get("authorization_code") or ""
        verifier = authorization.get("code_verifier") or ""
        if not auth_code or not verifier:
            raise RuntimeError("OpenAI returned an incomplete authorization response.")
        token = sdk.net.http_json(
            TOKEN_URL,
            method="POST",
            body=_form({
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": f"{ISSUER}/deviceauth/callback",
                "client_id": CLIENT_ID,
                "code_verifier": verifier,
            }),
            headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
        )
        if token.get("status") == 429:
            raise RuntimeError("OpenAI rate-limited the token exchange. Wait and try again.")
        if token.get("status") != 200:
            raise RuntimeError(f"Token exchange failed with HTTP {token.get('status', 0)}.")
        payload = token.get("body") or {}
        access = payload.get("access_token") or ""
        refresh = payload.get("refresh_token") or ""
        if not access or not refresh:
            raise RuntimeError("OpenAI did not return both access and refresh tokens.")
        now = datetime.now(timezone.utc)
        state = {
            "access_token": access,
            "refresh_token": refresh,
            "last_refresh": now.isoformat().replace("+00:00", "Z"),
            "source": "device-code",
        }
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            state["expires_at"] = now.timestamp() + float(expires_in)
        return state
