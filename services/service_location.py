"""Service plugin providing ambient location context to the agent.

While loaded, the service contributes a short "## Location" block to the
agent system prompt via ``agent_prompt_for``. A manually configured
location always wins; otherwise the service resolves an approximate
location by IP geolocation (keyless, stdlib urllib only) and caches it.

The prompt contribution lands in the semi-stable (cacheable) block of the
system prompt, so the returned string must stay stable between turns: the
prompt path never performs network I/O — reads serve the cache, and a
stale cache triggers a background refresh that takes effect next turn.
"""

dependencies_files = []
dependencies_pip = []

import json
import logging
import threading
import time
import urllib.request

from plugins.BaseService import BaseService, MANAGED

logger = logging.getLogger("LocationService")


class LocationService(BaseService):
    """Resolves and caches the user's approximate location for the agent prompt."""

    model_name = "location"
    shared = True
    lifecycle = MANAGED
    load_timeout = 30.0
    # Location sharing is consent-gated: the service ships unloaded (managed
    # services only autoload when the user adds them to autoload_services),
    # and while unloaded it still tells the agent the capability is off.
    prompt_when_unloaded = True

    config_settings = [
        ("Manual Location", "location_manual",
         "Your location as free text (e.g. \"Seattle, WA\" or a full address). "
         "When set, it is used as-is and no network lookup is performed.",
         "",
         {"type": "text"}),

        ("Location Refresh (minutes)", "location_refresh_minutes",
         "How long an automatically detected (IP-based) location stays fresh "
         "before it is re-resolved in the background.",
         60,
         {"type": "slider", "range": (5, 1440, 287), "is_float": False}),
    ]

    # Keyless IP-geolocation endpoints, tried in order. Each maps the raw
    # JSON payload onto the shared field names used by _format_auto.
    PROVIDERS = [
        ("https://ipinfo.io/json",
         lambda d: {"city": d.get("city"), "region": d.get("region"),
                    "country": d.get("country"), "timezone": d.get("timezone")}),
        ("http://ip-api.com/json",
         lambda d: {"city": d.get("city"), "region": d.get("regionName"),
                    "country": d.get("country"), "timezone": d.get("timezone")}
         if d.get("status") != "fail" else {}),
    ]

    def __init__(self, config: dict | None = None):
        """Initialize the location service."""
        super().__init__()
        self.config = config or {}
        self._cache: dict = {}
        self._fetched_at: float = 0.0
        self._refresh_lock = threading.Lock()
        self._refreshing = False

    # ── Lifecycle ───────────────────────────────────────────────────

    def _load(self) -> bool:
        """Mark loaded and warm the cache in the background (never blocks boot)."""
        self.loaded = True
        if not self._manual():
            self._refresh_async()
        return True

    def unload(self):
        """Release the cached location so a reload starts fresh."""
        self.loaded = False
        self._cache = {}
        self._fetched_at = 0.0

    # ── Agent prompt contribution ───────────────────────────────────

    def agent_prompt_for(self, ctx) -> str:
        """The '## Location' block, or '' when nothing is known yet.

        Cache-only: serves the last resolved location and kicks a
        background refresh when stale, so the semi-stable prompt block
        never waits on the network and stays byte-stable between turns.

        While unloaded (the shipped default — location sharing is opt-in),
        the block instead tells the agent the service is off so it can point
        the user at loading it when a question needs their location.
        """
        if not self.loaded:
            return ("## Location\nThe location service is installed but "
                    "unloaded, so the user's location is not shared. If the "
                    "user asks something location-dependent, tell them they "
                    "can load the location service (e.g. via /services) to "
                    "share it.")
        manual = self._manual(getattr(ctx, "config", None))
        if manual:
            return f"## Location\nThe user's location (user-provided): {manual}"
        if self._stale():
            self._refresh_async()
        text = self._format_auto(self._cache)
        if not text:
            return ""
        return ("## Location\nThe user's approximate location (IP-based, "
                f"may lag travel/VPNs): {text}")

    # ── Internals ───────────────────────────────────────────────────

    def _manual(self, config: dict | None = None) -> str:
        """The manually configured location, preferring live config when given."""
        cfg = config if isinstance(config, dict) and config else self.config
        return str((cfg or {}).get("location_manual") or "").strip()

    def _ttl_seconds(self) -> float:
        """Handle ttl seconds."""
        try:
            minutes = float(self.config.get("location_refresh_minutes") or 60)
        except (TypeError, ValueError):
            minutes = 60
        return max(minutes, 1) * 60

    def _stale(self) -> bool:
        """Handle stale."""
        return (time.time() - self._fetched_at) > self._ttl_seconds()

    def _refresh_async(self):
        """Kick one background refresh; concurrent calls coalesce."""
        with self._refresh_lock:
            if self._refreshing:
                return
            self._refreshing = True
        threading.Thread(target=self._refresh, daemon=True,
                         name="location-refresh").start()

    def _refresh(self):
        """Resolve the location once; failures degrade to the previous cache."""
        try:
            for url, extract in self.PROVIDERS:
                data = self._fetch_json(url)
                fields = extract(data) if isinstance(data, dict) else {}
                fields = {k: str(v).strip() for k, v in (fields or {}).items() if v}
                if fields:
                    self._cache = fields
                    break
            # A failed sweep still stamps the clock so we don't hammer
            # providers every prompt build while offline.
            self._fetched_at = time.time()
        finally:
            with self._refresh_lock:
                self._refreshing = False

    def _fetch_json(self, url: str) -> dict:
        """Fetch one provider payload; any failure returns {}."""
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "SecondBrain-Location/1.0",
                              "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=8) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception as e:
            logger.debug(f"Location provider failed ({url}): {e}")
            return {}

    @staticmethod
    def _format_auto(fields: dict) -> str:
        """Render cached fields as 'City, Region, Country (timezone TZ)'."""
        place = ", ".join(fields[k] for k in ("city", "region", "country") if fields.get(k))
        tz = fields.get("timezone") or ""
        if place and tz:
            return f"{place} (timezone {tz})"
        return place or (f"timezone {tz}" if tz else "")


def build_services(config: dict) -> dict:
    """Build services."""
    return {"location": LocationService(config)}
