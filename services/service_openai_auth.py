"""OpenAI subscription auth — a credential this app uses and never holds.

``llm_openai`` talks to a ChatGPT *subscription* endpoint rather than to
``platform.openai.com``, so what it needs is not an API key but an OAuth token
pair with a lifetime: obtained once by a device-code flow, refreshed
thereafter, and persisted across restarts. This service owns that pair. The
backend never sees it.

WHY THIS IS A SERVICE AND NOT PART OF THE BACKEND
-------------------------------------------------
Both halves of the token's life happen with nobody watching. The first write
lands whenever the user finishes signing in on their phone; the refresh lands
mid-conversation, possibly inside a scheduled subagent. ``config.write`` on an
unattended chain is UNSAFE and would be *refused*, not asked.

The escape is the policy's ownership mechanism: ``policy._owns_setting``
resolves a setting's declarer through ``plugin_discovery`` — "plugin names
(any kind)", services included — so a plugin writing a setting it declared in
its own ``config_settings`` is SAFE whoever is or is not present. Declaring the
three credential settings below is therefore not bookkeeping; it is the entire
reason this file exists as a service. A backend could not do it: backends are
not a plugin family and own nothing in the setting registry.

WHY SIGN-IN IS ON THE POLL THREAD
---------------------------------
Same argument ``service_drive`` makes, and its docstring says it best: the
sign-in is here "purely for *which thread it is on*". ``start`` runs on the
boot thread and owns a deadline, so a sign-in that waits on a human would time
the box out and take the boot with it. ``poll`` runs on a thread the kernel
starts afterwards, so the app finishes booting and the frontends come up while
the user is still reaching for their phone.

WHY DEVICE CODE AND NOT A LOCAL BROWSER
---------------------------------------
``service_drive`` uses ``run_local_server``, which binds a port and opens a
browser *on the host*. That is fine on a laptop and useless on a headless box
running a chat frontend — the browser opens where nobody is sitting. A device
code is a URL and a short string, so it can be completed from any device, and
its polling maps onto the poll loop instead of blocking a tick.

WHY THE ENDPOINTS ARE CONFIGURATION
-----------------------------------
Nothing about the flow is hardcoded: client id, both URLs, and the scopes are
settings the user fills in. That follows ``service_drive``, which requires a
``credentials.json`` the user brings from the Google console rather than
shipping OAuth client details in the package — and it means this file stays
correct when an endpoint moves, instead of shipping a constant that silently
starts returning 404.

WHAT NEVER CROSSES
------------------
``token`` hands back the ``<secret:...>`` handle, never plaintext. The kernel
substitutes the real value inside ``sdk.net.http``, so the backend places an
authenticated call holding a credential it was never given, and cannot leak
one it never had. The refresh below does the same to itself.
"""

import time

from guest.bases import BaseService

# How close to expiry is close enough to renew. Generous on purpose: the cost
# of refreshing early is one HTTP call, and the cost of refreshing late is a
# 401 in the middle of somebody's turn.
_RENEW_WITHIN = 300.0

# What the device-code endpoint says while it waits for the user. Neither is a
# failure, and treating them as one is the specific way this flow kills itself
# — see the note in ``poll``.
_PENDING = "authorization_pending"
_SLOW_DOWN = "slow_down"

# Sourced from OpenClaw's own OAuth documentation, which is where anyone
# setting this up is going to look anyway. Repeated as constants because
# ``config_settings`` has to be an AST-readable *literal* — a name reference
# there reads as nothing at all — and ``on_install`` needs the same values
# before the settings have been reconciled into config. Keep the two in step;
# the literals below are the ones a user sees, these are the fallback.
_DEFAULTS = {
    "openai_oauth_exchange_url": "https://auth.openai.com/oauth/token",
    "openai_oauth_scopes": "openid profile email offline_access",
}


def _host_of(url):
    """The host part of a URL, lowercased, or "" if there is none.

    Hand-rolled rather than ``urllib.parse`` because this needs three string
    operations and the guest's import surface is not the place to spend a
    dependency on a package whose neighbours open sockets.
    """
    text = str(url or "").strip()
    if "//" in text:
        text = text.split("//", 1)[1]
    return text.split("/", 1)[0].split("?", 1)[0].lower()


class OpenAIAuth(BaseService):
    """Holds a ChatGPT subscription token pair, and renews it."""

    name = "openai_auth"
    description = "OAuth tokens for the ChatGPT subscription backend."

    exports = ["token", "status", "sign_out"]
    requests = ["config.read", "config.write", "net.http", "session.push"]

    # Short, and deliberately not ``service_drive``'s 3600. Its sign-in blocks
    # inside a single tick, so one tick is all it needs; a device-code flow is
    # the opposite shape — it needs a tick per poll of the exchange endpoint,
    # and the endpoint itself asks for an interval in seconds. Once signed in
    # the tick is an integer comparison against the clock, so a short interval
    # costs nothing and buys renewal that happens before a call needs it
    # rather than during one.
    poll_interval = 5.0

    config_settings = [
        ("OpenAI OAuth client id", "openai_oauth_client_id",
         "The OAuth client this app identifies itself as when signing in to "
         "your ChatGPT subscription. Supplied by you, like Drive's "
         "credentials.json — nothing is shipped in the package.",
         "", {"type": "text"}),
        ("OpenAI device-code URL", "openai_oauth_device_url",
         "Endpoint that issues a device code and a user code.",
         "", {"type": "text"}),
        ("OpenAI exchange URL", "openai_oauth_exchange_url",
         "Endpoint that exchanges a device code for tokens, and later "
         "exchanges a refresh token for a new access token. Fixed "
         "infrastructure rather than anything of yours, so it is seeded.",
         "https://auth.openai.com/oauth/token", {"type": "text"}),
        ("OpenAI OAuth scopes", "openai_oauth_scopes",
         "Space-separated scopes to request during sign-in. Seeded with the "
         "set OpenClaw documents; offline_access is the one that matters, "
         "since without it no refresh token is issued and you would be "
         "signing in again every hour.",
         "openid profile email offline_access", {"type": "text"}),
        ("OpenAI responses URL", "openai_responses_url",
         "Inference endpoint llm_openai posts to. Read by the backend, "
         "declared here so it is configured in one place with the auth it "
         "needs.",
         "", {"type": "text"}),

        # Written by this service, never by a person. Declaring them is what
        # makes the unattended write SAFE; hiding them keeps /config honest
        # about which settings are actually yours to edit.
        ("OpenAI access token", "secret_openai_oauth_access",
         "Held by openai_auth. Reads back as a handle, never plaintext.",
         "", {"type": "text", "hidden": True}),
        ("OpenAI refresh token", "secret_openai_oauth_refresh",
         "Held by openai_auth. Reads back as a handle, never plaintext.",
         "", {"type": "text", "hidden": True}),
        ("OpenAI token expiry", "openai_oauth_expires_at",
         "Unix time the access token stops working.",
         0, {"type": "number", "hidden": True}),
    ]

    # ── install ─────────────────────────────────────────────────────

    def on_install(self, sdk):
        """Allowlist the endpoints this service has been pointed at.

        Egress is gated per *host* against ``net_allowed_hosts``, and that
        list is config a person maintains rather than something a plugin may
        declare about itself — contained code does not get to widen its own
        reach. ``/packages`` is the one moment where a plugin can ask: the
        chain roots at the command the user typed, so each write raises one
        dialog naming the setting and the value.

        Nothing is derived when the URLs are still blank, which is the usual
        case on a *first* install — they are configured afterwards. That gap
        is covered at runtime by ``_post``, which recognises the refusal and
        says which host to add. On a reinstall or an update the URLs are
        already there and this seeds them properly.

        Read-then-skip: this runs again on every update whose bytes changed,
        and a list the user has edited since is theirs.
        """
        hosts = [host for host in (
            _host_of(sdk.config.read(key) or _DEFAULTS.get(key))
            for key in ("openai_oauth_device_url",
                        "openai_oauth_exchange_url",
                        "openai_responses_url")) if host]
        allowed = sdk.config.read("net_allowed_hosts") or []
        missing = [host for host in dict.fromkeys(hosts)
                   if host not in allowed]
        if missing:
            sdk.config.write("net_allowed_hosts", [*allowed, *missing])

    def on_uninstall(self, sdk):
        """Take the credential with the package.

        Leaving a live subscription token in config after removing the only
        thing that could use it is the kind of leftover nobody goes looking
        for. The allowlist entries go too, since this service is what put
        them there — unlike, say, a folder the user was already indexing.
        """
        for key in ("secret_openai_oauth_access",
                    "secret_openai_oauth_refresh"):
            sdk.config.write(key, "")
        sdk.config.write("openai_oauth_expires_at", 0)

        hosts = {host for host in (
            _host_of(sdk.config.read(key) or _DEFAULTS.get(key))
            for key in ("openai_oauth_device_url",
                        "openai_oauth_exchange_url",
                        "openai_responses_url")) if host}
        allowed = sdk.config.read("net_allowed_hosts") or []
        keep = [host for host in allowed if host not in hosts]
        if len(keep) != len(allowed):
            sdk.config.write("net_allowed_hosts", keep)

    def __init__(self):
        """Nothing is acquired until start()."""
        # The device-code grant currently being waited on, or None. Lives on
        # the instance rather than in config because it is worthless after a
        # restart: the code expires in minutes and the next poll issues a
        # fresh one, so persisting it would only mean waking up to poll a
        # grant that died while the app was down.
        self._device = None
        self._told_them = False

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Return immediately, whether or not there is a token.

        Deliberately does not sign in and does not refresh. Having no token is
        not a failure — it is the honest state *installed, not yet signed in*,
        and reporting it as a failure would stop the service loading, which
        would in turn stop the poll that is the only way back to signed in.
        That is the trap ``service_drive`` documents at its own refresh branch:
        the one recoverable failure was the one that locked the door.
        """
        if not self._configured(sdk):
            sdk.log("openai_auth is installed but not configured; set "
                    "openai_oauth_client_id and the two URLs in /config",
                    level="warning")
        elif self._authorized(sdk):
            sdk.log("openai subscription token loaded")
        else:
            sdk.log("openai_auth is configured but not signed in; the first "
                    "poll will start the device-code flow")
        return True

    def poll(self, sdk):
        """Drive sign-in and renewal. Never raises, never blocks.

        Returns falsy on every path: there is no queue to drain here, so the
        loop waits ``poll_interval`` between ticks and the device-code
        interval is honoured by counting ticks rather than by sleeping inside
        one. Sleeping would be charged against this box's deadline, which is
        the wrong place to spend it.

        **Nothing here may raise.** A raising poll counts against
        ``max_poll_failures``, and five stop the loop for the life of the
        process. That is a real hazard rather than a theoretical one for this
        flow specifically: the exchange endpoint answers
        ``authorization_pending`` on *every* tick until the user finishes, so
        a version of this that treated a non-200 as an error would reliably
        kill itself on the fifth tick of a sign-in nobody had got round to
        yet. ``sdk.net.http`` helps by returning error statuses as answers
        rather than raising, so the pending case is an ordinary branch.
        """
        try:
            if not self._configured(sdk):
                return False
            if self._authorized(sdk):
                self._device = None
                self._told_them = False
                if self._expiring(sdk):
                    self._renew(sdk)
                return False
            if self._has_refresh(sdk):
                # A refresh token with no usable access token beside it.
                # Renew from it rather than starting a fresh sign-in: this is
                # the state after a restart that lost the access token, and
                # it is also how somebody arrives who signed in elsewhere and
                # pasted the refresh token into /config by hand. Renewal does
                # not care how the credential was first obtained, and a failed
                # renewal signs out, which lands on the device flow next tick
                # anyway.
                self._renew(sdk)
                return False
            if self._device is None:
                self._begin(sdk)
            else:
                self._collect(sdk)
        except Exception as exc:                             # noqa: BLE001
            # Swallowed for the reason above, and reported so the silence is
            # not total. The user hears about it once per load rather than
            # every five seconds — a notification storm about an unreachable
            # endpoint is worse than the outage.
            sdk.log(f"openai sign-in tick failed: {exc}", level="error")
            if not self._told_them:
                self._told_them = True
                self._notify(sdk, "OpenAI sign-in is not working",
                             f"The sign-in could not proceed: {exc}\n\n"
                             "Check the endpoints in `/config`, then reload "
                             "**openai_auth** from `/services`.",
                             level="error")
        return False

    def stop(self, sdk):
        """Drop the pending grant. Tokens live in config and survive."""
        self._device = None
        return True

    # ── exports ─────────────────────────────────────────────────────

    def token(self, sdk):
        """The access token, as a handle the caller can use but not read.

        Answers a shape rather than a bare string so the backend can tell
        "not signed in yet" from "here it is" without inspecting a sentinel.
        A caller that gets ``authorized: False`` should fail its call and say
        why — never wait, because what it would be waiting for is a person.

        The lazy renewal is a backstop, not the main path. ``poll`` renews
        ahead of expiry precisely so a live model call never pays for a token
        round trip; this branch only fires if the poll thread has stopped.
        """
        if not self._configured(sdk):
            return {"authorized": False,
                    "detail": "openai_auth is not configured"}
        if not self._authorized(sdk):
            return {"authorized": False,
                    "detail": "not signed in — check your notifications"}
        if self._expiring(sdk):
            self._renew(sdk)
        return {"authorized": True,
                "token": sdk.config.read("secret_openai_oauth_access"),
                "url": sdk.config.read("openai_responses_url") or ""}

    def status(self, sdk):
        """What state this is in, for ``/services`` and for a person asking.

        ``loaded`` and ``authorized`` come apart here for the same reason they
        do in ``service_drive``: installed-but-not-signed-in is a real and
        common state, and collapsing the two would report a service that
        cannot answer as healthy.
        """
        return {
            "loaded": True,
            "configured": self._configured(sdk),
            "authorized": self._authorized(sdk),
            "awaiting_user": self._device is not None,
            "expires_at": self._expires_at(sdk),
        }

    def sign_out(self, sdk):
        """Forget the tokens. The next poll starts a fresh device flow."""
        for key in ("secret_openai_oauth_access",
                    "secret_openai_oauth_refresh"):
            sdk.config.write(key, "")
        sdk.config.write("openai_oauth_expires_at", 0)
        self._device = None
        self._told_them = False
        sdk.log("openai subscription signed out")
        return {"authorized": False}

    # ── the flow ────────────────────────────────────────────────────

    def _post(self, sdk, url, payload):
        """POST JSON, turning an egress refusal into something actionable.

        A host missing from ``net_allowed_hosts`` is *refused* here rather
        than asked, because none of this runs with a person present — and a
        bare "denied" would leave the user with a sign-in that silently never
        happens. Naming the host and the setting is the difference between a
        five-second fix and an evening.
        """
        try:
            return sdk.net.http_json(url, method="POST", json=payload)
        except sdk.Denied as exc:
            raise RuntimeError(
                f"egress to {_host_of(url) or url} was refused. Add it to "
                f"net_allowed_hosts in /config, then reload openai_auth "
                f"from /services.") from exc

    def _begin(self, sdk):
        """Ask for a device code and tell the user where to type it."""
        device_url = sdk.config.read("openai_oauth_device_url")
        if not device_url:
            # Checked here rather than in ``_configured`` because this is the
            # only thing that needs it. Raising reaches ``poll``'s handler,
            # which notifies once per load rather than every five seconds.
            raise RuntimeError(
                "no device-code endpoint configured. Either set "
                "openai_oauth_device_url, or sign in with another client and "
                "paste its refresh token into secret_openai_oauth_refresh.")
        answer = self._post(
            sdk, device_url,
            {"client_id": sdk.config.read("openai_oauth_client_id"),
             "scope": sdk.config.read("openai_oauth_scopes") or ""},
        )
        body = answer.get("body") or {}
        if answer.get("status", 0) >= 400 or not body.get("device_code"):
            raise RuntimeError(
                f"device-code request returned {answer.get('status')}: "
                f"{body.get('error_description') or body.get('error') or body}")

        # The endpoint's own pacing, honoured in ticks. Its ``interval`` is
        # advisory until it says ``slow_down``, at which point it is not.
        interval = float(body.get("interval") or 5)
        self._device = {
            "device_code": body["device_code"],
            "interval": max(interval, self.poll_interval),
            "next_try": time.time(),
            "expires_at": time.time() + float(body.get("expires_in") or 600),
        }

        url = (body.get("verification_uri_complete")
               or body.get("verification_uri") or "")
        code = body.get("user_code") or ""
        self._notify(
            sdk, "Sign in to OpenAI",
            f"Second Brain needs to connect to your ChatGPT subscription.\n\n"
            f"Open **{url}** and enter the code **{code}**.\n\n"
            f"You can do this on any device — it does not have to be the "
            f"machine Second Brain is running on. This code expires in a few "
            f"minutes; if you miss it, another one is issued automatically.",
            level="info")

    def _collect(self, sdk):
        """Poll the exchange endpoint for the grant the user is completing."""
        grant = self._device
        now = time.time()
        if now < grant["next_try"]:
            return
        if now >= grant["expires_at"]:
            # Not an error and not worth a notification: the next tick issues
            # a fresh code and notifies with that one, which is the only thing
            # a person could act on anyway.
            sdk.log("openai device code expired before it was used")
            self._device = None
            return
        grant["next_try"] = now + grant["interval"]

        answer = self._post(
            sdk, sdk.config.read("openai_oauth_exchange_url"),
            {"client_id": sdk.config.read("openai_oauth_client_id"),
             "device_code": grant["device_code"],
             "grant_type": "urn:ietf:params:oauth:grant-type:device_code"},
        )
        body = answer.get("body") or {}
        error = body.get("error") or ""

        if error == _PENDING:
            return                       # the ordinary case, most ticks
        if error == _SLOW_DOWN:
            grant["interval"] += 5.0
            return
        if answer.get("status", 0) >= 400 or not body.get("access_token"):
            self._device = None
            raise RuntimeError(
                f"sign-in failed: "
                f"{body.get('error_description') or error or body}")

        self._store(sdk, body)
        self._device = None
        self._told_them = False
        sdk.log("openai subscription authorized")
        self._notify(sdk, "OpenAI is connected",
                     "Second Brain can now use your ChatGPT subscription. "
                     "Point a model profile at the **llm_openai** backend in "
                     "`/llm` if you have not already.",
                     level="info")

    def _renew(self, sdk):
        """Exchange the refresh token for a new access token.

        The refresh token goes out as its handle, in the request *body* — the
        kernel's substitution covers url, params, headers and body alike, so
        this renews a credential it cannot read. Note that this only holds
        while the body is JSON: form-encoding the handle would percent-escape
        the angle brackets, the substitution regex would no longer match, and
        the literal string ``<secret:...>`` would be sent to OpenAI. If this
        ever needs to be ``application/x-www-form-urlencoded``, the handle has
        to be spliced in unencoded.
        """
        answer = self._post(
            sdk, sdk.config.read("openai_oauth_exchange_url"),
            {"client_id": sdk.config.read("openai_oauth_client_id"),
             "refresh_token": sdk.config.read("secret_openai_oauth_refresh"),
             "grant_type": "refresh_token"},
        )
        body = answer.get("body") or {}
        if answer.get("status", 0) >= 400 or not body.get("access_token"):
            # The refresh token is dead. Clear the pair rather than leaving a
            # token that cannot be renewed: an empty credential makes the next
            # poll start a fresh device flow, where a stale one would fail
            # every call forever while reporting itself as signed in.
            sdk.log("openai refresh failed; signing out to re-authorize",
                    level="warning")
            self.sign_out(sdk)
            self._notify(sdk, "OpenAI needs signing in again",
                         "The saved credential could not be renewed. A new "
                         "sign-in code is on its way.",
                         level="warning")
            return
        self._store(sdk, body)
        sdk.log("openai subscription token renewed")

    def _store(self, sdk, body):
        """Persist a token payload.

        Every one of these writes is SAFE only because this service declared
        the settings — see the module docstring. Nothing here is attended.

        The refresh token is written only when the payload carries one, since
        an endpoint that does not rotate them omits the field and blanking it
        would lose the one credential that can renew the others.
        """
        sdk.config.write("secret_openai_oauth_access", body["access_token"])
        if body.get("refresh_token"):
            sdk.config.write("secret_openai_oauth_refresh",
                             body["refresh_token"])
        sdk.config.write(
            "openai_oauth_expires_at",
            int(time.time() + float(body.get("expires_in") or 3600)))

    # ── small facts ─────────────────────────────────────────────────

    def _configured(self, sdk):
        """Whether there is enough to do *anything*.

        Deliberately does not require ``openai_oauth_device_url``: that one is
        needed only to *start* a sign-in here, and a user who signed in
        elsewhere and pasted their refresh token into ``/config`` has a fully
        working setup without it. Requiring it would refuse to renew a
        perfectly good credential over a missing endpoint nothing was going to
        call. ``_begin`` checks for it at the point it actually needs it.
        """
        return all(sdk.config.read(key, present=True) for key in (
            "openai_oauth_client_id", "openai_oauth_exchange_url"))

    def _has_refresh(self, sdk):
        """Whether a refresh token exists, however it got here."""
        return bool(sdk.config.read("secret_openai_oauth_refresh",
                                    present=True))

    def _authorized(self, sdk):
        """Whether a usable access token exists.

        ``present=True`` answers with a bool and reveals nothing, which is
        what makes this askable at all — reading the setting itself would only
        ever hand back a handle, and a handle looks identical whether or not
        there is anything behind it.
        """
        return bool(sdk.config.read("secret_openai_oauth_access",
                                    present=True))

    def _expires_at(self, sdk):
        """When the access token dies. 0 when unknown."""
        try:
            return float(sdk.config.read("openai_oauth_expires_at") or 0)
        except (TypeError, ValueError):
            return 0.0

    def _expiring(self, sdk):
        """Whether the token is close enough to expiry to renew now.

        An unknown expiry counts as expiring: renewing a token that had years
        left costs one call, and *not* renewing one that died costs the turn.
        """
        expires = self._expires_at(sdk)
        return expires <= 0 or (expires - time.time()) <= _RENEW_WITHIN

    def _notify(self, sdk, title, body, *, level):
        """Tell the user something, tolerating there being nobody to tell.

        Guarded for the same reason ``service_drive`` guards its own: the poll
        thread can start before the runtime is fully up, and a notification
        that cannot be delivered must not become the exception that ends the
        sign-in. What is on the line is a message about a message.
        """
        try:
            sdk.session.push(body, title=title, notify=True, level=level)
        except Exception as exc:                             # noqa: BLE001
            sdk.log(f"could not notify ({title}): {exc}", level="debug")
