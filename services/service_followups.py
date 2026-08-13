"""The doorman that makes the agent offer follow-up chips, and hides the tool.

Half of a pair with ``tool_suggest_followups``, which it carries as a dependency
— the tool is useless without something to demand it, so installing this brings
both and uninstalling takes both away.

**Why a doorman rather than a tool the agent may reach for.** An offered tool is
a tool the model decides about, and a model deciding whether to bother is a
model that mostly does not: the same suggestion, offered, gets taken on some
turns and not others, and the row of chips flickers in and out for reasons
nobody can see. ``RequireTool`` removes the decision. What stays optional is the
*content* — an empty list is a real answer, and the note below says so — which
is the part actually worth the model's judgement.

**Why it is hidden the rest of the time.** ``suggest_followups`` is meaningful
at exactly one moment, and a tool in the catalogue is a tool in every prompt:
paid for on every turn, and available to be called at moments where it means
nothing. So ``shape_scope`` takes it out of the toolbox the model is shown, and
the doorman puts it back for one pinned call. That pairing is why
``runtime/conversation_loop.py`` looks past visibility when resolving a required
tool — before that, hiding a tool broke the very doorman that needed it, and
broke it silently.

**Web sessions only.** Chips are drawn by ``frontend_http``; a Telegram or
console session has nowhere to put them. Requiring the tool there would buy a
model round trip per turn and throw the answer away, so the guard is a plain
prefix check on the session key — free, and on the drive thread that matters.

Nothing here does real work. The whole file is guards and one note, because a
hook runs synchronously inside every turn it touches.
"""

dependencies_files = ['tools/tool_suggest_followups.py']
dependencies_pip = []
requests = ["config.read"]

from guest.bases import BaseService
from guest.hooks import RequireTool

#: The tool this doorman demands. Must match ``name`` on the tool class; the
#: kernel resolves it by name and an unknown one logs and lets the turn end.
TOOL_NAME = "suggest_followups"

#: Only sessions whose key starts with this can draw the chips. ``frontend_http``
#: keys its sessions ``http:<thread>``.
WEB_SESSION_PREFIX = "http:"


class Followups(BaseService):
    """Require follow-up suggestions at the end of a web turn."""

    name = "followups"
    description = (
        "Asks the agent for short follow-up questions at the end of each web turn "
        "and shows them as pressable buttons under the reply."
    )
    hooks = {"shape_scope": "hide_tool", "end_turn": "require_followups"}
    exports = []

    config_settings = [
        ("Follow-up suggestions", "followups_enabled",
         "Show pressable follow-up questions under the agent's replies in the web "
         "client. Turn off to stop asking for them without uninstalling.",
         True,
         {"type": "boolean"}),
        ("Follow-up count", "followups_count",
         "How many follow-up buttons to ask for. One or two read as a suggestion; "
         "more reads as a menu. Set to 0 to stop asking.",
         1,
         {"type": "integer"}),
        ("Follow-ups may be declined", "followups_allow_decline",
         "Let the agent offer nothing when no follow-up genuinely helps. Turn off "
         "to insist on suggestions every turn, which produces filler on turns that "
         "have no natural next question.",
         True,
         {"type": "boolean"}),
    ]

    def start(self, sdk):
        """Nothing to hold — the hooks are the whole service."""
        return True

    # ── shape_scope ────────────────────────────────────────────────

    def hide_tool(self, sdk, ctx, scope):
        """Keep the tool out of the toolbox the model is shown.

        Returning the list unchanged would be the same as abstaining, but this
        runs on every model call of every turn, so it stays a single list
        comprehension and reads no config: a setting that turned *hiding* off
        would only ever put a tool in the prompt that the agent must not call
        on its own initiative.
        """
        return [name for name in scope.tools if name != TOOL_NAME]

    # ── end_turn ───────────────────────────────────────────────────

    def require_followups(self, sdk, ctx, ending):
        """Demand one call to the hidden tool before the agent may leave.

        Abstains far more often than it fires. The ``doorman_fires`` check is
        what makes this terminate: the required call ends the turn again, and
        the second time round the count is no longer zero, so we stand aside and
        the agent leaves. It counts *every* doorman rather than only this one,
        which is the conservative direction — another plugin's intervention
        makes us abstain rather than pile on.
        """
        if not str(getattr(ctx, "session_key", "") or "").startswith(WEB_SESSION_PREFIX):
            return None
        if getattr(ending, "doorman_fires", 0):
            return None
        count = self._count(sdk)
        if count <= 0 or not self._enabled(sdk):
            return None
        return RequireTool(TOOL_NAME, note=self._note(sdk, count))

    # ── settings ───────────────────────────────────────────────────

    @staticmethod
    def _enabled(sdk) -> bool:
        """Whether to ask at all. Unset reads as on."""
        try:
            return sdk.config.read("followups_enabled") is not False
        except Exception:
            return True

    @staticmethod
    def _count(sdk) -> int:
        """How many chips to ask for. The tool clamps this again on the way out."""
        try:
            return int(sdk.config.read("followups_count"))
        except Exception:
            return 1

    @staticmethod
    def _note(sdk, count: int) -> str:
        """What the model is told at the moment it is handed the tool.

        This is the only guidance that exists, and it deliberately lives here
        rather than in ``agent_prompt``: the tool is invisible on every ordinary
        turn, so prompt text describing it would be paid for constantly and
        would describe something the model cannot see. An ephemeral note on the
        forced call costs nothing until the moment it is true.
        """
        try:
            may_decline = sdk.config.read("followups_allow_decline") is not False
        except Exception:
            may_decline = True
        plural = "question" if count == 1 else "questions"
        note = [
            f"Before finishing, call '{TOOL_NAME}' with up to {count} short follow-up "
            f"{plural} the person might naturally ask next.",
            "Write each in their voice, as something they would say to you, in a few "
            "words — it is submitted verbatim as their next message when pressed.",
            "Do not repeat anything already asked or already answered in this "
            "conversation.",
        ]
        if may_decline:
            note.append(
                "If nothing genuinely useful follows from this turn, call it with an "
                "empty list rather than inventing something."
            )
        return " ".join(note)
