"""What a person may answer an approval dialog with, and what it remembers.

:mod:`sandbox.approval` answers *how* we ask. This answers *what may be said*,
and — the part that makes it worth its own module — it is the only code in the
sandbox that **writes config**. That is safe for exactly one reason: what it
writes is a person's own answer to a question the kernel asked. Guest code
never reaches this; a plugin cannot propose an option, and the dialog cannot
be reached without the policy having already refused the Request.

Every "allow once" used to be thrown away. The only durable grant was
``skip_permissions``, whose unit is a *plugin name* — trusting ``web_search``
trusted every host it would ever reach. The lists here are narrower and the
person edits them the same way they read them, with ``/config``, which is the
whole reason these live in config rather than in the database: an approval
nobody can enumerate is one nobody can withdraw.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("Sandbox")


@dataclass(frozen=True)
class Option:
    """One answer a person may give to an approval dialog.

    ``value`` crosses to the frontend, comes back through
    ``FormStep.match_enum``, and lands in the ``answer_approval`` ledger row —
    so it is written to be read back months later ("always:api.brave.com"),
    not to be short. ``label`` is the only part meant to be read.

    ``remember`` is the entire difference between an answer and a grant:
    ``callable() -> bool``, built by the builder with the request already in
    hand, answering whether it changed anything. The dialog never learns what
    it does, which is what lets a future option write somewhere else entirely —
    a turn-scoped store, a user setting, a deny list.
    """

    value: str
    label: str
    allow: bool = True
    remember: Callable[[], bool] | None = None


#: Always offered, and the only two that are.
ALLOW_ONCE = Option("allow", "Allow once")
DENY = Option("deny", "Deny", allow=False)


# ──────────────────────────────────────────────────────────────────────
# The seam.
# ──────────────────────────────────────────────────────────────────────
#
# ``builder(chain, request, decision) -> list[Option]``. Every builder is asked
# about every dialog and answers with options it can *make good on*, or
# nothing. Deliberately the same shape as ``policy._SHELL_RECOGNIZERS``, and
# for the same three reasons:
#
#   1. **Abstain, never assert.** A builder returns options or ``[]``. It has
#      no say in whether the Request is allowed — that was settled by
#      ``classify`` before anything here ran. A bug costs an option.
#   2. **A raiser abstains**, logged and skipped, so one bad builder cannot
#      take the dialog down with it.
#   3. **Never offer what cannot work.** An option writing to a list the policy
#      would then refuse to honour — a folder inside the app's own tree, say —
#      is worse than no option at all: it is a grant the person believes they
#      made. Builders check before offering.
#
# Adding "Approve for the rest of this turn", "Always trust this tool", "Deny
# forever" or a mode switch is an entry in this list and a function beside it.
# It is not a branch in ``build_approver``.
#
# The three that ship all do the same thing one layer down — they turn "yes"
# into an entry in a list the user keeps — so each is really only two
# questions: what is the grantable unit here, and is it already granted.


def _always_allow_host(chain, request, decision) -> list:
    """Offer to add this request's host to the egress allowlist.

    Host-exact, and deliberately not the registrable domain:
    ``policy._host_allowed`` already matches *downward* on a dot boundary, so
    offering ``brave.com`` from a dialog about ``api.search.brave.com`` would
    grant strictly more than the question asked about.
    """
    from . import policy
    from .guest import requests as R

    if request.type != R.NET_HTTP:
        return []
    host = policy.request_host(request.args.get("url"))
    if not host or policy._host_allowed(host, policy._allowed_hosts()):
        return []
    return [Option(f"always:{host}", f"Always allow {host}",
                   remember=lambda: remember("net_allowed_hosts", host))]


def _always_allow_folder(chain, request, decision) -> list:
    """Offer to add the folder each unwritable end of this request sits in.

    One option per *end*, because a move needs both to be freely writable
    before the dialog stops appearing — offering only the destination would
    hand back a grant that changes nothing.

    The folder is the target's parent: predictable, and never more than the
    write being looked at. Granting a parent later tidies the children away
    (:func:`_merge_dir`), so the list converges instead of sprawling.
    """
    from . import policy
    from .guest import requests as R

    if request.type not in (R.FS_WRITE, R.FS_WRITE_BYTES, R.FS_MOVE,
                            R.FS_DELETE):
        return []
    args = request.args
    ends = ([args.get("src"), args.get("dst")] if request.type == R.FS_MOVE
            else [args.get("path")])

    options, seen = [], set()
    for end in ends:
        if not end or policy._freely_writable(end):
            continue
        try:
            target = Path(end).expanduser().resolve()
        except (OSError, ValueError, RuntimeError):
            continue
        folder = target if target.is_dir() else target.parent
        # A filesystem root is not a grant anybody means to make.
        if folder == folder.parent:
            continue
        # Rule 3: never offer what cannot work. A folder inside the app's own
        # tree or ``installed/`` is one ``_freely_writable`` will keep
        # refusing, so a button here would be a grant the person believes they
        # made — the worst outcome available.
        if policy._protected(target) or policy._protected(folder):
            continue
        shown = str(folder)
        if shown in seen:
            continue
        seen.add(shown)
        options.append(Option(
            f"always:{shown}", f"Always allow {shown}",
            remember=lambda folder=shown: remember("fs_writable_dirs", folder)))
    return options


def _always_allow_command(chain, request, decision) -> list:
    """Offer to remember this command's ``(program, subcommand)`` prefix.

    ``policy.command_prefix`` decides what that is and answers "" when there
    is no unit it can describe honestly — a named shell, a metacharacter, a
    program reached by path. No unit, no option.
    """
    from . import policy
    from .guest import requests as R

    if request.type not in (R.PROC_RUN, R.PROC_START):
        return []
    prefix = policy.command_prefix(policy._exact_argv(request.args))
    if not prefix or prefix.casefold() in policy._allowed_prefixes():
        return []
    return [Option(f"always:{prefix}", f"Always allow: {prefix}",
                   remember=lambda: remember("shell_allowed_prefixes", prefix))]


OPTION_BUILDERS: list = [_always_allow_host, _always_allow_folder,
                         _always_allow_command]


def options_for(chain, request, decision) -> list:
    """Every answer this dialog may offer, in the order it should show them.

    Allow-once first, extras in registry order, deny last. Duplicates by value
    are dropped, as are options with an empty value or label — those would be
    filtered out of the ``enum`` by ``runtime_approvals._sane_enum`` anyway,
    and it is better to never build a button than to build one nobody can
    press.
    """
    options = [ALLOW_ONCE]
    for build in OPTION_BUILDERS:
        try:
            options.extend(build(chain, request, decision) or [])
        except Exception:
            logger.exception("approval option builder failed; skipping it")
    options.append(DENY)

    seen, kept = set(), []
    for option in options:
        if not option.value or not option.label or option.value in seen:
            continue
        seen.add(option.value)
        kept.append(option)
    return kept


def chosen(options, value):
    """The option a coerced answer names, or ``None``.

    ``None`` is a real outcome rather than a defensive branch: a restored
    session can be answered against a dialog whose options were built by an
    older version of this file, and the caller treats an unrecognised answer as
    a refusal.
    """
    if value is None:
        return None
    for option in options:
        if option.value == value:
            return option
    return None


# ──────────────────────────────────────────────────────────────────────
# Writing it down.
# ──────────────────────────────────────────────────────────────────────

#: Serializes the read-merge-write below. Two approvals settling at once is
#: ordinary — the gate hands unsafe Requests to a pool — and the classic lost
#: update is two hosts approved in the same second, one of which vanishes.
#: ``config_manager.save`` takes no lock of its own; this covers our half.
_WRITE_LOCK = threading.Lock()


def _entries(raw) -> list:
    """A setting's current value as a list of non-empty strings.

    Both existing lists accept a comma-separated *string* as well as a list
    (``policy._allowed_hosts`` and ``_writable_dirs`` each handle it), so a
    hand-edited config must not be silently discarded by the first grant made
    after it.
    """
    if isinstance(raw, str):
        raw = raw.split(",")
    return [text for item in (raw or []) if (text := str(item).strip())]


def _merge_text(entry: str, existing: list):
    """Exact, case-folded de-dupe. ``None`` when nothing would change."""
    if entry.casefold() in {item.casefold() for item in existing}:
        return None
    return existing + [entry]


def _merge_host(entry: str, existing: list):
    """Add a host, dropping any listed subdomain it now covers.

    ``policy._host_allowed`` already matches downward on a dot boundary, so
    listing ``example.com`` makes a listed ``api.example.com`` dead weight.
    """
    from .policy import _host_allowed

    host = entry.strip().lower().lstrip(".")
    if not host or _host_allowed(host, {item.strip().lower().lstrip(".")
                                        for item in existing}):
        return None
    kept = [item for item in existing
            if not _host_allowed(item.strip().lower().lstrip("."), {host})]
    return kept + [host]


def _merge_dir(entry: str, existing: list):
    """Add a folder, dropping any listed folder it now contains.

    Subsumption both ways: a folder already inside a listed one changes
    nothing, and granting a parent tidies away the children that made it
    necessary. That is what keeps the list readable after a week of clicking
    "always allow" on one project.
    """
    from .policy import _within

    if _within(entry, existing):
        return None
    try:
        resolved = str(Path(entry).expanduser().resolve())
    except (OSError, ValueError, RuntimeError):
        return None
    kept = [item for item in existing if not _within(item, [resolved])]
    return kept + [resolved]


#: setting -> how a new entry joins the list it lives in. The three lists have
#: three different notions of "already covered", and each one lives here rather
#: than in the builder that offers it.
MERGERS = {
    "net_allowed_hosts": _merge_host,
    "fs_writable_dirs": _merge_dir,
    "shell_allowed_prefixes": _merge_text,
}


def remember(key: str, entry: str) -> bool:
    """Add one entry to a kernel list setting. ``True`` if anything changed.

    Three things are load-bearing.

    **The live dict is mutated in place.** ``kernel_config()`` hands back the
    same object ``policy._allowed_hosts`` reads, so the grant takes effect
    before ``save`` returns — which matters, because the Request that raised
    this dialog is about to be re-decided against it.

    **An unwired kernel writes nothing.** ``kernel_config()`` answers with a
    fresh ``{}`` when the composition root has not built one; mutating that is
    a silent no-op, and ``save({})`` would write ``DEFAULTS`` over a real
    user's file.

    **A no-op saves nothing**, so re-granting something already covered adds no
    ledger row and pushes no "settings changed" notice into the chat.
    """
    merge = MERGERS.get(key)
    if merge is None:
        logger.error("no merger for setting %s; refusing to write it", key)
        return False
    try:
        from runtime.context import kernel_config
    except Exception:
        logger.exception("no kernel to remember %s in", key)
        return False

    config = kernel_config()
    if not config:
        return False

    with _WRITE_LOCK:
        merged = merge(entry, _entries(config.get(key)))
        if merged is None:
            return False
        previous = config.get(key)
        config[key] = merged
        try:
            from config import config_manager

            config_manager.save(config)
        except Exception:
            # A widening that is live but unpersisted would come back as a
            # surprise at the next restart, in the direction nobody wants.
            config[key] = previous
            logger.exception("could not persist %s", key)
            return False
    logger.info("remembered %s in %s", entry, key)
    return True
