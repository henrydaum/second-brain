"""
SERVICE TEMPLATE
================
A service is a long-lived capability other plugins call: a loaded model, a
connection pool, a cache. Reference for authoring one; not imported by the
running system.

Read docs/SDK.md for the Request surface and sandbox/guest/bases.py for every
attribute a service can declare. This file covers what is specific to services.

Before writing: read docs/SDK.md, then this entire template. For details not
defined here, inspect sandbox/guest/bases.py (BaseService declarations),
sandbox/bridge.py (kernel adapter and exports), sandbox/guest/hooks.py (turn
hooks), and sandbox/guest/sdk.py (service calls and Requests). Validate the
finished file before registering it.

  Where it goes:  DATA_DIR/workspace/services/service_<name>.py
  Filename:       must start with "service_"
  Entry points:   start(self, sdk) and stop(self, sdk), plus its exports

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

  1. You get a subprocess automatically, and cannot decline one. Isolation is
     decided by the kernel from where the file lives — an installed package
     that imports a foreign library is subprocessed because the validator can
     see the import. There is nothing to declare.
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


STANDING AT A DOORWAY
---------------------
Services are also where hooks live, because something has to be resident for
the kernel to call into. Declare them and write the methods:

    hooks = {"end_turn": "check_done"}

See templates/hook_template.py for all six moments and their payloads.


LISTENING TO THE BUS
--------------------
Same shape, same reason: declare the channels and write one handler.

    subscribed_channels = ["task_completed"]

    def on_event(self, sdk, channel, payload):
        ...

A channel you did not declare is never delivered. Only services and frontends
can subscribe at all — a tool is a call that ends, so there would be nothing
left to deliver to.

Two things bite here. Handlers run **on the thread that emitted**, so a slow
on_event slows down whoever published — enqueue a task instead of doing real
work. And a channel name is just a string: the kernel's live in
events/event_channels.py, but plugins own their own, so nothing checks the
spelling and a typo is silence rather than an error.


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
    # Guidance added to the agent's system prompt while this service is
    # loaded, and gone when it is not. Use ``def agent_prompt(self, sdk)``
    # instead when the text depends on live state.
    agent_prompt = (
        "## Embeddings\n"
        "Ask the embedder service for vectors; do not compute them yourself."
    )

    # A foreign library does its own work, so the kernel puts a process around
    # this automatically — the import is what decides, not a declaration.
    dependencies_pip = ["sentence-transformers"]

    # The public surface. _model and _normalize stay internal.
    exports = ["embed", "similarity"]

    # Bus channels this service hears. Declared, so there is no subscription
    # to drop at unload and none can survive an uninstall.
    subscribed_channels = ["config_changed"]

    config_settings = [
        ("Embedding model", "embedder_model_name",
         "Which sentence-transformers model to load.",
         "all-MiniLM-L6-v2", {"type": "text"}),
    ]

    def on_install(self, sdk):
        """Arrange what this service needs, once, when its files are installed.

        Runs before this service loads, under the chain of the ``/packages``
        command the user typed — which is the only moment a plugin can write a
        kernel setting, because it is the only one where somebody is present to
        be asked. Each write raises one dialog naming the setting and the
        value; reads and SQL are free.

        Read-then-skip, because this runs again on every update that changes
        the file. A value the user has edited since is theirs.
        """
        cache = sdk.path.join(sdk.paths.get("workspace"), "embeddings")
        watched = sdk.config.read("sync_directories") or []
        if cache not in watched:
            sdk.config.write("sync_directories", [*watched, cache])

    def on_uninstall(self, sdk):
        """Undo what belongs to this service, before its files are removed.

        Still on disk, still registered, pip dependencies still installed —
        this is the first step of the uninstall, so all three are available.
        Raising is reported and the package goes anyway.

        Note what is *not* undone: the sync entry above. A table this plugin
        created is unambiguously its own; a folder the user has been indexing
        for months is not, whoever put it there first.
        """
        sdk.db.define("DROP TABLE IF EXISTS embedder_cache")

    def start(self, sdk):
        """Load the model once. Return True on success."""
        self._model = None
        self._load(sdk)
        return True

    def _load(self, sdk):
        """Load the model if it is not already held. Internal, so unexported."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_name = sdk.config.read("embedder_model_name")
            sdk.log(f"loading {model_name}")
            # Held on the instance, which lives in the box and never crosses out.
            self._model = SentenceTransformer(model_name)
        return self._model

    def stop(self, sdk):
        """Release the model. Must tolerate never having started."""
        self._model = None

    def on_event(self, sdk, channel, payload):
        """React to a declared channel. One handler for all of them.

        Note what this does *not* do: reload the model right here. The handler
        runs on the thread that emitted, so whoever saved the config would wait
        out a model load. Dropping a reference is the right amount of work.
        """
        if channel == "config_changed" and "embedder_model_name" in (
                payload.get("keys") or []):
            sdk.log("model setting changed; will reload on next use")
            self._model = None

    def embed(self, sdk, texts):
        """Return one vector per text. Simple data, so it can cross out."""
        model = self._load(sdk)
        return [list(map(float, v)) for v in model.encode(texts)]

    def similarity(self, sdk, a, b):
        """Cosine similarity between two texts.

        Arithmetic is not a Request. Nothing here touches disk, network, clock
        or process, so it runs right where it is written, at no cost and with
        nothing to approve — the SDK is for *effects*, and computation was
        never on the other side of the boundary.

        Note what this is not for. Comparing two things you already hold is
        this; ranking a table of stored vectors is a query, and it belongs in
        SQL over ``vec_cosine`` so the answer crosses the boundary instead of
        the corpus. See ``sdk.db.query`` in docs/SDK.md.
        """
        first, second = self.embed(sdk, [a, b])
        dot = sum(x * y for x, y in zip(first, second))
        norms = (sum(x * x for x in first) ** 0.5
                 * sum(y * y for y in second) ** 0.5)
        return dot / norms if norms else 0.0


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
