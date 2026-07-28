"""Every command that declares a consequential Request declares a gate for it.

A command's ``requests`` list is what one "yes" buys. The state machine is
what *asks*, and it only asks when the command declares ``require_approval``
or an ``approval_actions``/``approval_action_prefixes`` predicate. Declare the
capability and not the gate and the command runs ungranted: no dialog before
the body, and the Request falls through mid-run to the execution-time
approver.

That fallback path is real and now works, but it is the wrong shape for a
command — it interrupts a half-finished operation with a question about a
Request the user has no context for, where the up-front grant states the whole
scope before anything happens. ``/packages install`` shipped with exactly this
gap, and while it was open it deadlocked the process.

One test rather than one per command, and the consequential set is *derived*
from the policy rather than restated, so a Request added to ``ALWAYS_UNSAFE``
tomorrow is covered here without anyone remembering to come back.
"""

from plugins.helpers.plugin_paths import iter_plugin_dirs
from sandbox.guest.requests import NET_HTTP, PROC_RUN, SECRET_REVEAL
from sandbox.policy import ALWAYS_UNSAFE
from sandbox.validator import validate_file

# ``ALWAYS_UNSAFE`` plus the three whose branch is about arguments rather than
# about the caller — running a shell, reaching the network, and handing over a
# credential are consequential however they are spelled.
#
# ``config.write`` is deliberately absent: policy already makes a plugin
# persisting its *own* declared setting safe, so requiring a gate for it would
# demand approval declarations from commands that only ever write their own
# keys.
CONSEQUENTIAL = ALWAYS_UNSAFE | {PROC_RUN, NET_HTTP, SECRET_REVEAL}


def _command_files():
    """Every command source on this machine, across all three trees.

    Enumerated by path and read by AST — never imported. A store command may
    pull in a third-party library that has no business being loaded into the
    test process, which is the same reason ``package_manager`` reads package
    metadata this way.
    """
    for family, directory in iter_plugin_dirs():
        if family != "command" or not directory.exists():
            continue
        for path in sorted(directory.glob("command_*.py")):
            yield path


def test_every_consequential_command_declares_its_approval_gate():
    """A declared capability with no gate is a dialog that never renders."""
    ungated = []
    for path in _command_files():
        declared = validate_file(path).declarations
        if not declared:
            continue  # not a migrated command; the native contract differs
        asks_for = CONSEQUENTIAL & set(declared.get("requests") or ())
        if not asks_for:
            continue
        gated = (
            declared.get("require_approval")
            or declared.get("approval_actions")
            or declared.get("approval_action_prefixes")
        )
        if not gated:
            ungated.append(f"{path.name} declares {sorted(asks_for)}")

    # Collected rather than asserted one at a time, so a run names every
    # offender instead of stopping at the first.
    assert not ungated, (
        # ASCII on purpose: this lands on a Windows console under cp1252,
        # where a unicode dash raises instead of rendering.
        "these commands declare consequential Requests but nothing that makes "
        "the state machine ask for them up front. Add require_approval or "
        "approval_actions:\n  " + "\n  ".join(ungated)
    )
