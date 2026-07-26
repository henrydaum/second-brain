"""Secret handles — using a credential you are never allowed to read.

The database is not where the secrets are. ``users.password_hash`` is the only
secret column in the schema; the real ones are API keys and OAuth tokens in
config and the environment. So this is the mechanism the security contract's
"private information" clause actually needs, and it sits on ``config.read``
and ``env.read``, not on ``db.query``.

A Request that would return a credential returns an opaque handle instead::

    key = sdk.config.read("brave_api_key").data     # "<secret:brave_api_key>"
    sdk.net.http(url, headers={"X-Key": key})       # kernel swaps it back

Sandboxed code can therefore *use* a credential it can never *read*, which is
exactly the property a careless plugin needs: it cannot leak a key it was
never given, cannot log one by accident, and cannot include one in an error
message.

Substitution happens on the way out, inside the handler, after the policy
function has already decided. A handle is meaningless anywhere else — it is
just a string, and one that does not resolve is left exactly as it is rather
than being silently blanked, so a mistake looks like a mistake.
"""

from __future__ import annotations

import re

PREFIX = "<secret:"
SUFFIX = ">"

_HANDLE = re.compile(r"<secret:([A-Za-z0-9_.\-]+)>")

# Word parts that make a config key or environment variable a credential.
# Matched against the name's *parts* rather than as substrings, because
# substring matching redacts ``max_tokens`` for containing "token" — and a
# rule that redacts ordinary settings trains people to work around it.
#
# Still deliberately generous within that: a false positive costs a plugin
# the plaintext of something it probably should not have had, which is the
# safe direction to be wrong in.
SECRET_WORDS = {"key", "apikey", "secret", "token", "password", "passwd",
                "credential", "credentials", "auth"}

_SPLIT = re.compile(r"[^a-z0-9]+")


def is_secret(name: str) -> bool:
    """Whether a config key or variable name holds a credential."""
    parts = set(_SPLIT.split((name or "").lower()))
    return bool(parts & SECRET_WORDS)


def handle_for(name: str) -> str:
    """The opaque stand-in for a named secret."""
    return f"{PREFIX}{name}{SUFFIX}"


def looks_like_handle(value) -> bool:
    """Whether a value is a handle rather than a real credential."""
    return isinstance(value, str) and bool(_HANDLE.fullmatch(value))


def redact(name: str, value):
    """Return the value, or a handle if the name says it is a credential."""
    return handle_for(name) if is_secret(name) else value


def resolve(value, lookup):
    """Swap handles for real values, recursively, on the way out.

    ``lookup`` takes a secret's name and returns its value or None. Anything
    that does not resolve is left untouched: a handle that reaches a remote
    server as literal ``<secret:foo>`` is a visible bug, where silently
    sending an empty string is an invisible one.
    """
    if isinstance(value, str):
        def _swap(match):
            """Replace one handle if we know it."""
            found = lookup(match.group(1))
            return found if isinstance(found, str) else match.group(0)
        return _HANDLE.sub(_swap, value)
    if isinstance(value, list):
        return [resolve(v, lookup) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve(v, lookup) for v in value)
    if isinstance(value, dict):
        return {k: resolve(v, lookup) for k, v in value.items()}
    return value


def lookup_from(ctx):
    """Build a resolver over a kernel context's config and environment."""
    import os

    def _lookup(name: str):
        """Find a secret by name, config first."""
        config = getattr(ctx, "config", None) or {}
        if name in config and isinstance(config[name], str):
            return config[name]
        return os.environ.get(name)

    return _lookup
