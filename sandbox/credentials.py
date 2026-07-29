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

**Named ``credentials`` and not ``secrets`` on purpose.** A subprocess box runs
with ``sandbox/`` as its cwd, so every top-level module here is importable in
the child under its bare name — and a file called ``secrets.py`` shadowed the
stdlib ``secrets`` for the whole guest process. Nothing in the sandbox imports
it, which is why it went unnoticed; litellm does, and every model call died on
``module 'secrets' has no attribute 'token_hex'``. ``tests/
test_sandbox_guest_boundary.py`` now refuses any stdlib-shadowing name here.
"""

from __future__ import annotations

import base64
import json
import re

PREFIX = "<secret:"
SUFFIX = ">"

_HANDLE = re.compile(r"<secret:([A-Za-z0-9_.\-]+)>")

# The declaration is the name. A config setting holding a credential is
# called ``secret_something``, and that is the whole rule.
#
# Second Brain already declares things this way — ``tool_``, ``command_`` and
# ``service_`` prefixes are how discovery knows what a file is — so this is
# the same convention one level down. It also puts the fact where it is
# useful: reading ``sdk.config.read("secret_brave_key")`` in a plugin, you can
# see you are getting a handle without going to look anything up.
SECRET_PREFIX = "secret_"

# Word parts that *look* like a credential. Deliberately not policy: this is
# what the validator warns about at authoring time ("this looks like a secret
# and is not marked — rename it"), so a bad guess costs a developer one
# message rather than surprising anybody at runtime.
#
# It is still the rule for environment variables, because nothing declares
# those — no plugin owns ``OPENAI_API_KEY`` and its name was chosen by
# somebody else entirely.
SECRET_WORDS = {"key", "apikey", "secret", "token", "password", "passwd",
                "credential", "credentials", "auth"}

_SPLIT = re.compile(r"[^a-z0-9]+")


def looks_secret(name: str) -> bool:
    """Whether a name reads like a credential. A guess, and used as one."""
    parts = set(_SPLIT.split((name or "").lower()))
    return bool(parts & SECRET_WORDS)


def is_secret(name: str, *, guess: bool = False) -> bool:
    """Whether a name holds a credential.

    ``guess`` is for names nobody declared — environment variables — where
    the heuristic is all there is. For config settings it stays False: the
    prefix is the declaration, and an unmarked setting is not a secret.
    """
    lowered = (name or "").lower()
    if lowered.startswith(SECRET_PREFIX):
        return True
    return guess and looks_secret(name)


def handle_for(name: str) -> str:
    """The opaque stand-in for a named secret."""
    return f"{PREFIX}{name}{SUFFIX}"


def looks_like_handle(value) -> bool:
    """Whether a value is a handle rather than a real credential."""
    return isinstance(value, str) and bool(_HANDLE.fullmatch(value))


def redact(name: str, value, *, guess: bool = False):
    """Return the value, or a handle if the name says it is a credential."""
    return handle_for(name) if is_secret(name, guess=guess) else value


def redact_nested(name: str, value):
    """Redact credential-shaped leaves in a structured config value.

    The handle encodes its original config path, so it remains usable after
    guest code moves or rewrites the surrounding structure.
    """
    def walk(current, path):
        if isinstance(current, dict):
            return {
                key: walk(item, [*path, str(key)])
                for key, item in current.items()
            }
        if isinstance(current, list):
            return [
                walk(item, [*path, str(index)])
                for index, item in enumerate(current)
            ]
        leaf = path[-1] if path else name
        if len(path) == 1 and is_secret(leaf):
            return handle_for(leaf)
        if is_secret(leaf) or leaf == "llm_api_key":
            raw = json.dumps(path, separators=(",", ":")).encode()
            token = base64.urlsafe_b64encode(raw).decode().rstrip("=")
            return handle_for(f"path_{token}")
        return current

    return walk(value, [name])


def resolvable(name: str) -> bool:
    """Whether a handle naming this may be swapped for a real value.

    A handle is meant to stand for something the kernel *gave* the guest: a
    ``secret_*`` setting read back through ``config.read``, an environment
    variable the name heuristic caught, or an encoded ``path_`` token this
    module minted itself. Nothing checked that, so any name at all resolved —
    guest code could write ``<secret:AWS_SECRET_ACCESS_KEY>`` into a header,
    or ``<secret:db_path>`` into a URL, and the kernel would substitute a value
    it had never handed over. Substitution now only reaches names that could
    have been a handle in the first place.
    """
    return bool(name) and (name.startswith("path_")
                           or is_secret(name, guess=True))


def resolve(value, lookup):
    """Swap handles for real values, recursively, on the way out.

    ``lookup`` takes a secret's name and returns its value or None. Anything
    that does not resolve is left untouched: a handle that reaches a remote
    server as literal ``<secret:foo>`` is a visible bug, where silently
    sending an empty string is an invisible one.
    """
    if isinstance(value, str):
        def _swap(match):
            """Replace one handle if we know it, and may hand it over."""
            name = match.group(1)
            if not resolvable(name):
                return match.group(0)
            found = lookup(name)
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
        if name.startswith("path_"):
            try:
                token = name[5:]
                token += "=" * (-len(token) % 4)
                path = json.loads(
                    base64.urlsafe_b64decode(token.encode()).decode())
                value = config
                for part in path:
                    value = (
                        value[int(part)]
                        if isinstance(value, list)
                        else value[part]
                    )
                return value
            except (KeyError, IndexError, TypeError, ValueError):
                return None
        if name in config and isinstance(config[name], str):
            return config[name]
        return os.environ.get(name)

    return _lookup
