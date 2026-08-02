"""Provide ambient location context to the agent.

A manually configured location wins. Otherwise the service resolves an
approximate location by IP when it starts, trying two keyless providers. The
result becomes a stable ``## Location`` system-prompt contribution.

Sandbox prompt contributions are cached until a plugin reloads, so location
is deliberately resolved at startup rather than refreshed behind the cached
prompt. Reload the service after changing location or network/VPN context.
"""

dependencies_files = []
dependencies_pip = []

import json

from guest.bases import BaseService


class LocationService(BaseService):
    """Resolve the user's approximate location for the agent prompt."""

    name = "location"
    description = "Add a configured or IP-derived location to the agent prompt."
    shared = True
    timeout = 30
    requests = ["config.read", "net.http"]

    config_settings = [
        ("Manual Location", "location_manual",
         "Your location as free text (e.g. \"Seattle, WA\" or a full address). "
         "When set, it is used as-is and no network lookup is performed.",
         "",
         {"type": "text"}),
    ]

    PROVIDERS = [
        ("https://ipinfo.io/json", "ipinfo"),
        ("http://ip-api.com/json", "ip-api"),
    ]

    def __init__(self):
        self.manual = ""
        self.location = {}

    def start(self, sdk):
        """Read configuration and resolve location once."""
        self.manual = str(sdk.config.read("location_manual") or "").strip()
        self.location = {}
        if self.manual:
            return True

        for url, provider in self.PROVIDERS:
            data = self._fetch_json(sdk, url)
            fields = self._extract(provider, data)
            if fields:
                self.location = fields
                break
        if not self.location:
            sdk.log("could not resolve an approximate location", level="warning")
        return True

    def stop(self, sdk):
        """Discard cached location state."""
        self.manual = ""
        self.location = {}

    def agent_prompt(self, sdk):
        """Return the stable location prompt block, if location is known."""
        if self.manual:
            return f"## Location\nThe user's location (user-provided): {self.manual}"
        text = self._format_auto(self.location)
        if not text:
            return ""
        return (
            "## Location\nThe user's approximate location (IP-based, may lag "
            f"travel/VPNs): {text}"
        )

    def _fetch_json(self, sdk, url):
        """Fetch one provider payload; failures allow the fallback provider."""
        try:
            answer = sdk.net.http(
                url,
                headers={
                    "User-Agent": "SecondBrain-Location/1.0",
                    "Accept": "application/json",
                },
            )
            status = int(answer.get("status") or 0)
            if status >= 400:
                sdk.log(f"location provider returned HTTP {status}: {url}",
                        level="debug")
                return {}
            return json.loads(answer.get("body") or "{}")
        except (sdk.Failed, ValueError, TypeError) as error:
            sdk.log(f"location provider failed ({url}): {error}", level="debug")
            return {}

    @staticmethod
    def _extract(provider, data):
        """Normalize a provider payload into shared location fields."""
        if not isinstance(data, dict):
            return {}
        if provider == "ip-api":
            if data.get("status") == "fail":
                return {}
            raw = {
                "city": data.get("city"),
                "region": data.get("regionName"),
                "country": data.get("country"),
                "timezone": data.get("timezone"),
            }
        else:
            raw = {
                "city": data.get("city"),
                "region": data.get("region"),
                "country": data.get("country"),
                "timezone": data.get("timezone"),
            }
        return {key: str(value).strip() for key, value in raw.items() if value}

    @staticmethod
    def _format_auto(fields):
        """Render fields as ``City, Region, Country (timezone TZ)``."""
        place = ", ".join(
            fields[key] for key in ("city", "region", "country")
            if fields.get(key)
        )
        timezone = fields.get("timezone") or ""
        if place and timezone:
            return f"{place} (timezone {timezone})"
        return place or (f"timezone {timezone}" if timezone else "")
