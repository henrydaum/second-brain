"""
SERVICE TEMPLATE
================
A service is a long-lived capability other plugins call: a loaded model, a
connection pool, a cache. Reference for authoring one; not imported by the
running system.

Read SDK.md for the Request surface and sandbox/guest/bases.py for every
attribute a service can declare. This file covers what is specific to services.

  Where it goes:  DATA_DIR/sandbox_plugins/services/service_<name>.py
  Filename:       must start with "service_"
  Entry points:   start(self, sdk) and stop(self, sdk), plus its exports

  ┌────────────────────────────────────────────────────────────────────┐
  │ NOT LOADABLE YET. sandbox/bridge.py bridges tools, tasks, and      │
  │ commands; services and frontends still return None. The contract   │
  │ below is settled and correct to write against, but a sandboxed     │
  │ service will not load until the bridge grows a service branch.     │
  │ Until then, a service that must run today stays native — see       │
  │ MIGRATING_PLUGINS.md.                                              │
  └────────────────────────────────────────────────────────────────────┘


A SERVICE IS A PERSISTENT BOX
-----------------------------
Services are the natural persistent box, and the only family that is one by
default. The box is loaded once, keeps its state, and is serialized — one call
at a time. You do not manage this; it follows from the family.

The rule that shapes everything else: STATE STAYS INSIDE THE BOX. The model,
the connection, the cache — those never cross the boundary. Callers get simple
data back and the thing itself never leaves. If you find yourself wanting to
return a client object so the caller can use it, export a method that does the
work instead.

Note the two similar words. Native services declare
`lifecycle = "managed" | "extension"`, about *who loads them*. Sandboxed
plugins declare `lifetime = "ephemeral" | "persistent"`, about *whether the
box survives between calls*. Services set the second one for you.


EXPORTS ARE THE PUBLIC SURFACE
------------------------------
Only methods named in `exports` are reachable through sdk.services.call.
Everything else is internal. This is what makes "which service methods can a
plugin call?" answerable by reading the file instead of guessing.

    exports = ["embed", "similarity"]

Every exported method must return simple data — the same constraint as any
Request return value. Exports are read without importing the file, so the list
must be a literal.


FOREIGN LIBRARIES AND CREDENTIALS
---------------------------------
Services are where foreign libraries usually live, and the honest position is:
a library that does its own I/O cannot be mediated. Two consequences:

  1. Declare `isolation = "subprocess"`. The library's actions are past the
     kernel's reach, so put a process boundary around them.
  2. If it needs a credential, you genuinely need the plaintext, because there
     is no Request for the kernel to substitute a handle into.

Name any credential setting `secret_something` — that prefix IS the
declaration. It then reads back as a `<secret:name>` handle, which the kernel
substitutes inside sdk.net.http, so code uses a credential it never held.

When a foreign library needs the real value:

    key = sdk.secrets.reveal("secret_my_api_key")

A plugin reading a key it declared in its own `config_settings` is not asked —
configuring it was the consent, and prompting on every load would be exactly
the approval fatigue this design avoids. A DIFFERENT plugin reaching for that
same key does get a dialog. Once you hold plaintext you are responsible for it.


DRIVING WORK ON A SCHEDULE
--------------------------
A service never imports the orchestrator or the tasks it drives. It emits, and
whatever declared that channel fires:

    sdk.events.emit("schedule.tick.daily", {"source": "my_service"})

Prefer a task's `default_jobs` over a thread in a service — the timekeeper
already owns the clock.
"""

from guest.bases import BaseService


class Embedder(BaseService):
    """A loaded model held inside the box; callers get vectors, never the model."""

    name = "embedder"
    description = "Sentence embeddings for search and clustering."

    # A foreign library does its own work, so put a process around it.
    isolation = "subprocess"
    dependencies_pip = ["sentence-transformers"]

    # The public surface. _model and _normalize stay internal.
    exports = ["embed", "similarity"]

    config_settings = [
        ("Embedding model", "embedder_model_name",
         "Which sentence-transformers model to load.",
         "all-MiniLM-L6-v2", {"type": "text"}),
    ]

    def start(self, sdk):
        """Load the model once. Return True on success."""
        from sentence_transformers import SentenceTransformer

        model_name = sdk.config.read("embedder_model_name")
        sdk.log(f"loading {model_name}")
        # Held on the instance, which lives in the box and never crosses out.
        self._model = SentenceTransformer(model_name)
        return True

    def stop(self, sdk):
        """Release the model. Must tolerate never having started."""
        self._model = None

    def embed(self, sdk, texts):
        """Return one vector per text. Simple data, so it can cross out."""
        return [list(map(float, v)) for v in self._model.encode(texts)]

    def similarity(self, sdk, a, b):
        """Cosine similarity between two texts."""
        first, second = self.embed(sdk, [a, b])
        # A pure helper — no Request, no cost, runs right here.
        return sdk.text.cosine(first, second)


class Weather(BaseService):
    """The credential case, done the way that does not leak: a handle."""

    name = "weather"
    description = "Current conditions from a weather API."

    exports = ["current"]
    requests = ["net.http"]

    config_settings = [
        ("Weather API key", "secret_weather_api_key",
         "API key for the weather provider.", "", {"type": "text"}),
    ]

    def start(self, sdk):
        """Nothing to acquire — the key is fetched per call, as a handle."""
        return True

    def current(self, sdk, city):
        """Fetch conditions for a city."""
        # This is a <secret:...> handle, not the key. It is safe to hold, log,
        # and pass around; the kernel swaps in the real value on the way out
        # of sdk.net.http. No approval dialog, because this plugin declared
        # the setting itself.
        key = sdk.config.read("secret_weather_api_key")
        response = sdk.net.http(
            f"https://api.example.com/current?city={city}",
            headers={"Authorization": f"Bearer {key}"},
        )
        return response["body"]
