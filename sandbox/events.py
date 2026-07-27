"""The bus, inbound — the only place that knows both sides of a delivery.

``sdk.events.emit`` has always worked: sandboxed code can shout onto the bus,
because publishing is a Request like any other. Hearing back is the direction
that needed building, and it needed building differently — an event arrives
*at* a plugin rather than being asked for, so there is no Request to classify
and no return value to translate. What there is instead is a subscription, and
a subscription is a thing that can leak.

So it is **declared, not registered**, exactly as ``hooks`` are: a plugin lists
``subscribed_channels``, the bridge stands a listener at each one and removes
them at unload. The plugin never holds a subscription, so it cannot forget to
drop one, and uninstalling the file takes the declaration with it.

Two things this module owes the rest of the system:

**Payloads are projections.** ``bus.request`` enriches a payload with a live
``threading.Event`` and a result list (``events/event_bus.py``) so a subscriber
can answer synchronously. Neither can cross a process boundary, and a
sandboxed subscriber must not be able to satisfy a round trip it cannot be
trusted to complete — a box that hangs would hang the publisher. ``project``
drops them, along with anything else that will not serialize, and a sandboxed
subscriber therefore sees ``bus.request`` as an ordinary fire-and-forget event.

**Nothing here may raise.** ``EventBus.emit`` runs handlers on the publisher's
own thread and swallows what they raise; a listener that let an exception out
would be logged and ignored, but a listener that *blocked* would stall whoever
published. Delivery is best-effort at every layer, which is the same failure
policy the ledger has and for the same reason: an observer must never break
the thing it observes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("Sandbox")

# What a payload may be built from. Anything else is dropped rather than
# coerced: a plugin receiving ``"<Thread(worker, started)>"`` where it expected
# an object is worse off than one receiving nothing, because the second case
# is obvious and the first one is not.
_SCALARS = (str, int, float, bool, type(None))

# Keys ``bus.request`` adds for its synchronous round trip. Named rather than
# inferred, so the reason they are gone is readable at the point of removal.
_ROUND_TRIP_KEYS = frozenset({"reply", "result"})

# How deep a payload may nest before it is treated as unserializable. Bus
# payloads are flat by convention; the limit is here so a self-referential one
# cannot spin.
_MAX_DEPTH = 6


def project(payload, _depth: int = 0):
    """Reduce a bus payload to something that can cross into a box.

    Returns ``None`` for anything that cannot be represented, which callers
    treat as "nothing to deliver" rather than as an error.
    """
    if _depth > _MAX_DEPTH:
        return None
    if isinstance(payload, _SCALARS):
        return payload
    if isinstance(payload, dict):
        clean = {}
        for key, value in payload.items():
            if not isinstance(key, str) or key in _ROUND_TRIP_KEYS:
                continue
            reduced = project(value, _depth + 1)
            # A key whose value would not cross is dropped, not nulled: a
            # subscriber checking ``"db" in payload`` should find it absent
            # rather than present and useless.
            if reduced is not None or value is None:
                clean[key] = reduced
        return clean
    if isinstance(payload, (list, tuple, set)):
        reduced = [project(item, _depth + 1) for item in payload]
        return [item for item in reduced if item is not None]
    return None


def build_listener(plugin, channel: str, deliver):
    """A bus handler that carries one channel into a plugin's box.

    ``deliver(channel, payload)`` is what actually crosses — supplied by the
    bridge, which owns the box handle. Keeping it a parameter is what lets this
    module stay ignorant of boxes, and lets a test deliver into a list.
    """

    def listener(payload=None):
        """Deliver one event. Never raises, never blocks on an answer."""
        try:
            deliver(channel, project(payload))
        except Exception:
            # The bus would swallow this anyway; logging it here names the
            # plugin, which the bus cannot do.
            logger.exception("delivering %s to %s failed",
                             channel, getattr(plugin, "name", "?"))

    listener.__name__ = f"deliver_{channel}"
    listener.__doc__ = (f"Carry {channel!r} into "
                        f"{getattr(plugin, 'name', '?')}'s box.")
    return listener


def subscribe_all(plugin, channels, deliver) -> list:
    """Stand a listener at every declared channel. Returns unsubscribers.

    The unsubscribe callables are the *only* handle on these subscriptions —
    ``EventBus.subscribe`` hands one back and keeps no other index — so the
    caller must hold them for as long as the plugin is loaded.
    """
    from events.event_bus import bus

    dropped = []
    for channel in channels or []:
        if not isinstance(channel, str) or not channel.strip():
            logger.warning("%s declared an unusable channel %r; skipping",
                           getattr(plugin, "name", "?"), channel)
            continue
        dropped.append(bus.subscribe(channel,
                                     build_listener(plugin, channel, deliver)))
    if dropped:
        logger.info("%s is listening on %s", getattr(plugin, "name", "?"),
                    ", ".join(sorted(c for c in channels if isinstance(c, str))))
    return dropped


def unsubscribe_all(unsubscribers) -> None:
    """Step away from every channel.

    A subscription outliving its plugin is a leak with no symptom — the box is
    gone, so every delivery fails quietly forever — which is why this tolerates
    anything rather than stopping at the first failure.
    """
    for drop in unsubscribers or []:
        try:
            drop()
        except Exception:
            logger.exception("could not unsubscribe a sandboxed listener")
