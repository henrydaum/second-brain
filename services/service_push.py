"""Device push notifications for scheduled agents, and nothing else.

Second Brain already tells you everything it does. The panel, the banner and the
bell are the right surface for almost all of it — you find out when you next
look, which is soon enough for a plugin registering or a setting changing. One
case is not like that: a scheduled agent reporting back. Asking for the news at
07:00 and being told at 11:00, when you happen to open the app, is not the thing
that was asked for. This service exists for that case and refuses the rest.

WHAT QUALIFIES, AND WHY IT CAN BE DECIDED HERE
----------------------------------------------
Two populations reach this handler and both are already unambiguous:

  ``source == "subagents"``  is *only ever* a scheduled agent failing. Its two
      emit sites in ``runtime/subagents.py`` are "Scheduled agent … failed" and
      "Scheduled agent did not start"; an interactive spawn's failure goes back
      to the model that spawned it and never lands here.

  ``source == "session"``    is a background turn's final answer, raised by
      ``emit_fallback_push`` only when the conversation's ``notification_mode``
      is ``on`` *and* nobody was attending the session. That is the right shape
      already, but it also covers a subagent spawned by hand — so the
      conversation's category is checked, because ``_scheduled_category`` files
      a timekeeper-fired child under "Scheduled" and a hand-spawned one under
      "Subagent".

Everything else — the plugin watcher, config announcements, compaction progress
— is dropped. A lock screen is a scarce surface and the whole value of this
plugin is that it stays quiet.

THE HONEST PART ABOUT ISOLATION
-------------------------------
Web Push bodies are AES128GCM-encrypted binary, and ``sdk.net.http`` decodes
text and cannot send bytes — ``sandbox/handlers/fs_net.py`` says so, and says
binary egress is absent on purpose. So this drives ``pywebpush``, which performs
its own network I/O. That has a consequence worth stating plainly rather than
burying: **this plugin loads disclaimed and runs in a subprocess, and the
kernel cannot mediate the request that leaves this machine.** It is the
documented shape for a library that does its own I/O (docs/SDK.md, "Foreign
libraries and credentials"), not a way around the rule.

The VAPID private key is therefore genuinely revealed rather than passed as a
handle: there is no Request for the kernel to substitute one into.

BEFORE IT WORKS
---------------
1. Generate a key pair once — ``npx web-push generate-vapid-keys``, or
   ``python -m py_vapid --gen``.
2. Put the public key in ``push_vapid_public_key`` and the private key in
   ``secret_push_vapid_private_key``.
3. Put a real address in ``push_contact_email``. It becomes the VAPID ``sub``
   claim, which is how a push service reaches an operator whose subscriptions
   are misbehaving; Apple and Mozilla both reject a request without one.
4. Subscribe a device from the Second Brain UI: Settings, "Notify this device".
   The browser half lives in ``src/lib/push.ts`` in the UI repository and
   reaches ``subscribe`` below through the ordinary ``service.call`` Request, so
   nothing had to be added to the HTTP frontend or to Caddy for it.

``net_allowed_hosts`` is *not* consulted here — pywebpush bypasses
``sdk.net.http`` entirely — so adding ``web.push.apple.com`` there changes
nothing. It is mentioned only to save the next person the experiment.
"""

dependencies_files = []
dependencies_pip = ["pywebpush"]

import json
import time

from guest.bases import BaseService

#: Conversation categories whose background results are worth a device push.
#: ``runtime/subagents.py:_scheduled_category`` writes exactly these two.
#:
#: **Do not reference this name inside ``config_settings``.** Declarations are
#: extracted from the AST without importing the file, so every element of that
#: list has to be a literal. A single name reference in one default value makes
#: the whole list unextractable — and it is dropped in silence, taking every
#: other setting in it along. That failure shipped once: all four settings read
#: back ``null``, ``public_key()`` answered "", the browser subscribed against
#: an empty application server key, and the only symptom was a phone that never
#: buzzed. The literal below is duplicated for exactly that reason.
DEFAULT_CATEGORIES = ["Scheduled", "Scheduled (one-time)"]

#: A push service will reject an oversized payload outright, and a notification
#: body is a model's whole final answer — tables, code fences and all. The
#: budget is ~4KB after encryption; this leaves room for the rest of the JSON
#: and for the fact that a lock screen shows perhaps three lines anyway.
BODY_LIMIT = 800

#: Per-endpoint send timeout. Short on purpose: this runs on a bus thread, and
#: a push service that is not answering is not worth waiting for.
SEND_TIMEOUT = 10

#: How many consecutive failures before an endpoint is dropped. A 404 or 410 is
#: dropped immediately — that is the push service saying the subscription is
#: dead — so this only governs the ambiguous failures: timeouts, 500s, a laptop
#: that was asleep.
MAX_FAILURES = 10


class Push(BaseService):
    """Send scheduled-agent notifications to subscribed browsers."""

    name = "push"
    description = "Sends scheduled agent results to subscribed devices as push notifications."
    shared = True
    subscribed_channels = ["notification_pushed"]
    exports = ["public_key", "subscribe", "unsubscribe", "state", "send_test"]
    # No ``conv.read``: the category comes from ``db.query`` instead, for the
    # reason ``_category`` gives. Asking for a Request this never makes would be
    # capability nobody audits.
    requests = ["db.define", "db.query", "db.write",
                "config.read", "secret.reveal"]
    dependencies_pip = ["pywebpush"]

    config_settings = [
        ("VAPID public key", "push_vapid_public_key",
         "Base64url VAPID public key. Handed to the browser when it subscribes; "
         "generate a pair with `npx web-push generate-vapid-keys`.",
         "",
         {"type": "text"}),

        ("VAPID private key", "secret_push_vapid_private_key",
         "Base64url VAPID private key, the pair of the public key above. Stored "
         "as a secret. This one is revealed in plaintext rather than passed as a "
         "handle, because pywebpush signs with it directly and there is no "
         "Request for the kernel to substitute into.",
         "",
         {"type": "text"}),

        ("Push contact address", "push_contact_email",
         "An address the push service can reach you at, used as the VAPID `sub` "
         "claim. Apple and Mozilla reject pushes without one.",
         "",
         {"type": "text"}),

        ("Push categories", "push_categories",
         "Conversation categories whose background results are pushed to your "
         "devices, as a JSON array. Defaults to the two the timekeeper files "
         "scheduled agents under; empty means those defaults.",
         # A literal, never ``DEFAULT_CATEGORIES`` — see the note on that name.
         ["Scheduled", "Scheduled (one-time)"],
         {"type": "json_list"}),
    ]

    # ── lifecycle ───────────────────────────────────────────────────

    def on_install(self, sdk):
        """The subscriptions table.

        Keyed on the endpoint because that *is* the identity of a subscription:
        a browser that re-subscribes with the same keys produces the same URL,
        which is what makes ``subscribe`` an upsert and therefore safe to call
        on every launch of the app.
        """
        self._define(sdk)

    def on_uninstall(self, sdk):
        """Take the table with us. Nothing else here is ours."""
        try:
            sdk.db.define("DROP TABLE IF EXISTS push_subscriptions")
        except sdk.Failed as error:
            sdk.log(f"could not drop push_subscriptions: {error}", "warning")

    def start(self, sdk):
        """Nothing to open. The table is defined here as well as in
        ``on_install`` because a service can be registered from the sandbox
        without an install ever having run, and a missing table would then be
        discovered at the worst moment — the first scheduled agent to finish."""
        self._define(sdk)
        return True

    def stop(self, sdk):
        """No connection, no thread, nothing to tear down."""
        return None

    def _define(self, sdk):
        try:
            sdk.db.define(
                "CREATE TABLE IF NOT EXISTS push_subscriptions ("
                " endpoint TEXT PRIMARY KEY,"
                " p256dh TEXT NOT NULL,"
                " auth TEXT NOT NULL,"
                " label TEXT,"
                " created_at REAL,"
                " last_ok REAL,"
                " failures INTEGER DEFAULT 0)")
        except sdk.Failed as error:
            sdk.log(f"could not create push_subscriptions: {error}", "warning")

    # ── the browser's side ──────────────────────────────────────────

    def public_key(self, sdk):
        """The VAPID public key, for ``pushManager.subscribe``.

        Read at subscribe time rather than compiled into the frontend bundle, so
        rotating the pair is a config edit and a re-subscribe rather than a
        rebuild and a redeploy.
        """
        return str(sdk.config.read("push_vapid_public_key") or "").strip()

    def subscribe(self, sdk, endpoint="", keys=None, label=""):
        """Remember a browser. Idempotent on ``endpoint``.

        Called both when the user turns the toggle on and on every launch after
        that (``refreshPush``), because Safari rotates subscriptions without
        reliably firing ``pushsubscriptionchange``. Re-posting an unchanged
        subscription has to be free, and an upsert is how.

        The failure counter resets here: a device saying "I am here" is better
        evidence than a run of timeouts from when it was switched off.
        """
        keys = keys or {}
        endpoint = str(endpoint or "").strip()
        p256dh = str(keys.get("p256dh") or "").strip()
        auth = str(keys.get("auth") or "").strip()
        if not endpoint or not p256dh or not auth:
            return {"ok": False, "error": "endpoint and keys are required"}

        sdk.db.write(
            "INSERT INTO push_subscriptions"
            " (endpoint, p256dh, auth, label, created_at, last_ok, failures)"
            " VALUES (?, ?, ?, ?, ?, NULL, 0)"
            " ON CONFLICT(endpoint) DO UPDATE SET"
            "  p256dh = excluded.p256dh,"
            "  auth = excluded.auth,"
            "  label = excluded.label,"
            "  failures = 0",
            [endpoint, p256dh, auth, str(label or ""), time.time()])
        return {"ok": True}

    def unsubscribe(self, sdk, endpoint=""):
        """Forget a browser."""
        endpoint = str(endpoint or "").strip()
        if not endpoint:
            return {"ok": False, "error": "endpoint is required"}
        sdk.db.write("DELETE FROM push_subscriptions WHERE endpoint = ?",
                     [endpoint])
        return {"ok": True}

    def state(self, sdk, endpoint=""):
        """Whether this endpoint is known, and how many devices there are.

        The UI reads the *browser* for its toggle state, deliberately — that is
        where the truth is once someone revokes permission in iOS Settings. This
        exists for the other question, asked from a REPL: is anything actually
        subscribed?
        """
        rows = sdk.db.query(
            "SELECT endpoint, label, created_at, last_ok, failures"
            " FROM push_subscriptions ORDER BY created_at")
        endpoint = str(endpoint or "").strip()
        return {
            "configured": bool(self.public_key(sdk)),
            "count": len(rows),
            "known": any(row.get("endpoint") == endpoint for row in rows),
            "devices": [
                {"label": row.get("label") or "Browser",
                 "created_at": row.get("created_at"),
                 "last_ok": row.get("last_ok"),
                 "failures": row.get("failures")}
                for row in rows
            ],
        }

    def send_test(self, sdk, title="Second Brain", body="Test notification."):
        """Push something now, to every subscribed device.

        Worth having: the real path fires at 07:00 tomorrow, and finding out
        then that the key pair was mismatched is a poor way to spend a morning.
        """
        return self._deliver(sdk, {"title": title, "body": body,
                                   "sent_at": time.time()})

    # ── the bus ─────────────────────────────────────────────────────

    def on_event(self, sdk, channel, payload):
        """One notification. Push it only if a scheduled job produced it.

        Runs on a bus thread rather than the emitting one — ``sandbox/events``
        keeps delivery off the publisher — so the sends below are made inline.
        They are still bounded by ``SEND_TIMEOUT`` each, because "off the
        publisher's thread" is not the same as "free".

        **Must not raise.** The bus is fire-and-forget and a notification
        failing to become a push must never disturb the notification itself.
        """
        if channel != "notification_pushed":
            return
        payload = payload or {}
        try:
            if not self._qualifies(sdk, payload):
                return
            self._deliver(sdk, payload)
        except Exception as error:
            sdk.log(f"could not push a notification: {error}", "warning")

    def _qualifies(self, sdk, payload) -> bool:
        """Whether this notification is worth waking a phone for."""
        source = str(payload.get("source") or "")

        # Scheduled-agent failures, which is all this source ever carries. A job
        # that did not run is exactly as worth knowing as one that did.
        if source == "subagents":
            return True

        # A background turn's final answer. Real, but it covers hand-spawned
        # subagents too, and those are spawned by somebody who is sitting there.
        if source != "session":
            return False

        conversation_id = payload.get("conversation_id")
        if not conversation_id:
            return False

        category = self._category(sdk, conversation_id)
        if category is not None:
            return category in self._categories(sdk)

        # **The category could not be read.** Falling back rather than dropping,
        # because the alternative is this plugin going quiet in a way nobody
        # would notice for weeks. The fallback is narrow and safe: an
        # interactive spawn is created with ``notification_mode="off"``
        # (``runtime/subagents.py``), and ``emit_fallback_push`` only fires when
        # the mode is ``on`` — so a ``source: "session"`` notification from a
        # subagent session has already been filtered to scheduled ones by the
        # kernel. What this fallback lets through that the category check would
        # not is an ordinary conversation somebody deliberately turned
        # notifications on for, and only while the read is broken.
        return str(payload.get("source_id") or "").startswith("spawn_subagent:")

    def _categories(self, sdk) -> list:
        """Categories that qualify, from config, falling back to the defaults."""
        try:
            configured = sdk.config.read("push_categories")
        except sdk.Failed:
            return list(DEFAULT_CATEGORIES)
        if isinstance(configured, str):
            configured = [part.strip() for part in configured.split(",")]
        names = [str(name).strip() for name in (configured or []) if str(name).strip()]
        return names or list(DEFAULT_CATEGORIES)

    def _category(self, sdk, conversation_id):
        """One conversation's category, or ``None`` if it could not be read.

        ``db.query`` rather than ``conv.read``: that Request returns every
        message in the conversation, and a recurring job is pinned to one
        conversation precisely so its transcript accumulates. Reading a whole
        year of a daily digest to learn one string would get slower every day
        this plugin worked correctly.

        **``my_conversations``, never ``conversations``.** The bare name holds
        other people's rows and ``scope_sql`` refuses it outright
        (``sandbox/users.py``); the ``my_`` name expands to a subquery filtered
        to the calling user. Getting this wrong does not raise anything a user
        would see — it raises ``Denied`` in here, which is why the two outcomes
        below are distinguished rather than both answering "" and quietly
        meaning "not scheduled".

        The empty string and ``None`` are different answers: a conversation with
        no category is an ordinary chat and does not qualify, whereas a failed
        read is not evidence of anything. See ``_qualifies`` for what it does
        with the difference.
        """
        try:
            rows = sdk.db.query(
                "SELECT category FROM my_conversations WHERE id = ?",
                [int(conversation_id)], max_rows=1)
        except (TypeError, ValueError):
            return None
        except sdk.Failed as error:
            # Includes ``Denied``, which is what a scoping mistake or a chain
            # with no user arrives as. Logged at warning because the plugin is
            # now guessing, and a guess that lasts is a bug nobody is looking
            # for.
            sdk.log(f"could not read the category of conversation "
                    f"{conversation_id}: {error}", "warning")
            return None
        if not rows:
            return None
        return str(rows[0].get("category") or "").strip()

    # ── sending ─────────────────────────────────────────────────────

    def _deliver(self, sdk, payload) -> dict:
        """Encrypt and send to every subscribed device.

        Answers a small report rather than nothing, so ``send_test`` is useful
        from a REPL and a failure in the scheduled path leaves something in the
        log worth reading.
        """
        from pywebpush import WebPushException, webpush

        private_key = str(sdk.secrets.reveal("secret_push_vapid_private_key") or "").strip()
        contact = str(sdk.config.read("push_contact_email") or "").strip()
        if not private_key or not contact:
            sdk.log("push is not configured: set secret_push_vapid_private_key "
                    "and push_contact_email", "warning")
            return {"sent": 0, "failed": 0, "pruned": 0,
                    "error": "not configured"}

        subscriptions = sdk.db.query(
            "SELECT endpoint, p256dh, auth, failures FROM push_subscriptions")
        if not subscriptions:
            return {"sent": 0, "failed": 0, "pruned": 0}

        message = self._message(payload)
        claims = {"sub": contact if contact.startswith("mailto:")
                  else f"mailto:{contact}"}

        sent = failed = pruned = 0
        for row in subscriptions:
            endpoint = row.get("endpoint")
            try:
                webpush(
                    subscription_info={
                        "endpoint": endpoint,
                        "keys": {"p256dh": row.get("p256dh"),
                                 "auth": row.get("auth")},
                    },
                    data=message,
                    vapid_private_key=private_key,
                    vapid_claims=dict(claims),
                    ttl=86400,
                    timeout=SEND_TIMEOUT,
                    headers={"Urgency": "normal"},
                )
            except WebPushException as error:
                status = getattr(getattr(error, "response", None), "status_code", 0)
                # The push service's way of saying this subscription is dead:
                # the app was deleted, or permission was revoked. Keeping the
                # row would mean retrying forever against a URL that will never
                # work again.
                if status in (404, 410):
                    self._forget(sdk, endpoint)
                    pruned += 1
                else:
                    failed += 1
                    self._penalise(sdk, endpoint, row)
                continue
            except Exception as error:
                # A foreign library's own failure modes are not enumerable from
                # here, and one bad endpoint must not stop the others.
                sdk.log(f"push to {endpoint} failed: {error}", "warning")
                failed += 1
                self._penalise(sdk, endpoint, row)
                continue

            sent += 1
            sdk.db.write(
                "UPDATE push_subscriptions SET last_ok = ?, failures = 0"
                " WHERE endpoint = ?", [time.time(), endpoint])

        return {"sent": sent, "failed": failed, "pruned": pruned}

    def _message(self, payload) -> str:
        """The JSON `sw.js` will read.

        ``conversation_id`` is the payload's reason for existing beyond the
        text: it is what makes tapping the notification land in the conversation
        the scheduled agent wrote into, rather than wherever the app was last.
        """
        body = str(payload.get("body") or "").strip()
        if len(body) > BODY_LIMIT:
            body = body[:BODY_LIMIT].rstrip() + "…"
        return json.dumps({
            "title": str(payload.get("title") or "Second Brain").strip(),
            "body": body,
            "level": payload.get("level") or "info",
            "conversation_id": payload.get("conversation_id"),
            "notification_id": payload.get("notification_id"),
            "sent_at": payload.get("sent_at") or time.time(),
        })

    def _forget(self, sdk, endpoint) -> None:
        try:
            sdk.db.write("DELETE FROM push_subscriptions WHERE endpoint = ?",
                         [endpoint])
        except sdk.Failed as error:
            sdk.log(f"could not prune a dead subscription: {error}", "warning")

    def _penalise(self, sdk, endpoint, row) -> None:
        """Count an ambiguous failure, and drop an endpoint that only fails.

        Separate from the 404/410 path because these are recoverable: a phone in
        a tunnel, a push service having a bad minute. What is not recoverable is
        an endpoint that has failed ten times running, and keeping it costs a
        timeout on every scheduled agent from now until somebody notices.
        """
        failures = int(row.get("failures") or 0) + 1
        try:
            if failures >= MAX_FAILURES:
                sdk.db.write(
                    "DELETE FROM push_subscriptions WHERE endpoint = ?",
                    [endpoint])
                sdk.log(f"dropped a push endpoint after {failures} failures",
                        "warning")
            else:
                sdk.db.write(
                    "UPDATE push_subscriptions SET failures = ?"
                    " WHERE endpoint = ?", [failures, endpoint])
        except sdk.Failed:
            pass    # bookkeeping; the send already failed and was counted
