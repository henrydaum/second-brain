"""Offer the person one or two follow-up questions as pressable chips.

Half of a pair. ``service_followups`` is the doorman that demands this tool at
the end of a web turn and hides it the rest of the time; this is the half that
turns what the agent wrote into a frame the client can draw.

**The agent authors the chips as this tool's arguments**, which is the whole
point of the shape. The conversation is already in the model's context at the
moment it is asked, so there is no second model call, no separate prompt, and
nothing to re-send — the cost is the handful of output tokens the suggestions
themselves occupy. Generating them from a fresh call would mean paying for the
conversation twice to answer a question the model had just finished thinking
about.

**It leaves by the event bus, not by returning.** A tool's return value goes to
the model; chips have to reach the *client*, and only a loaded frontend may
render. So this emits on ``followups`` and ``frontend_http`` — which holds the
browser's stream — pushes the ``buttons`` frame. Nothing else listens, and a
session whose frontend cannot draw buttons simply has nobody to hear it, which
is why this is safe to call from any session.

**An empty list is a real answer.** "Nothing here is worth asking next" is a
judgement worth honouring, and it still emits — an empty ``buttons`` payload is
what clears chips left over from a previous turn.
"""

dependencies_files = []
dependencies_pip = []
requests = ["session.get", "event.emit", "config.read"]

from guest.bases import BaseTool

#: The bus channel ``frontend_http`` listens on, and must match the literal in
#: its ``subscribed_channels`` exactly. Channel names are not a closed
#: vocabulary and nothing validates either end, so a typo here is silence rather
#: than an error: chips would simply never appear, with no failure to find.
#: Change one spelling and you must change the other.
FOLLOWUPS_CHANNEL = "followups"

#: Hard ceiling on how many chips may be emitted, whatever the setting says.
#: A row of chips is a glance, not a menu; past a handful it stops being one.
MAX_SUGGESTIONS = 5

#: Longest a single chip may be. Chips are laid out in a row and wrap, so an
#: essay in one does not render as an essay — it renders as a broken row.
MAX_CHARS = 80


class SuggestFollowups(BaseTool):
    """Suggest follow-ups."""

    name = "suggest_followups"
    description = (
        "Offer the person short follow-up questions as pressable buttons under your "
        "reply. Each suggestion is submitted verbatim as their next message when "
        "pressed, so write it in their voice, as something they would ask you — not "
        "as an instruction to yourself. Keep each to a few words. Pass an empty list "
        "when nothing genuinely useful follows from this turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Follow-up questions, in the person's voice, a few words each. Empty means nothing is worth suggesting.",
            },
        },
        "required": ["suggestions"],
    }
    requires_services = []

    def run(self, sdk, **kwargs):
        """Run suggest followups."""
        raw = kwargs.get("suggestions")
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            return sdk.fail("'suggestions' must be a list of strings.")

        try:
            session = sdk.session.get() or {}
            session_key = session.get("key")
            if not session_key:
                # Nothing to address the frame to. Not a failure worth telling
                # the model about — it did its part — but there is no client.
                return sdk.ok({"emitted": 0}, llm_summary="No session to show buttons in.")

            buttons = self._clean(raw, self._limit(sdk))
            sdk.events.emit(FOLLOWUPS_CHANNEL, {
                "session_key": session_key,
                "buttons": buttons,
            })
        except Exception as error:
            return sdk.fail(f"Could not offer follow-ups: {error}")

        if not buttons:
            return sdk.ok({"emitted": 0}, llm_summary="No follow-ups offered.")
        shown = ", ".join(button["label"] for button in buttons)
        return sdk.ok(
            {"emitted": len(buttons), "buttons": buttons},
            llm_summary=f"Offered {len(buttons)} follow-up(s): {shown}",
        )

    @staticmethod
    def _limit(sdk) -> int:
        """How many chips the person has asked for, clamped to something sane."""
        try:
            configured = sdk.config.read("followups_count")
        except Exception:
            configured = None
        try:
            count = int(configured)
        except (TypeError, ValueError):
            count = 1
        return max(0, min(count, MAX_SUGGESTIONS))

    @staticmethod
    def _clean(raw, limit: int):
        """Trim, drop blanks and repeats, and cap — in the model's order.

        Deduplication is case-insensitive because two chips differing only in
        capitalisation read as a mistake, and the model has no way to know what
        it already wrote once the list is long.
        """
        buttons = []
        seen = set()
        for item in raw:
            text = str(item or "").strip()
            if not text:
                continue
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS].rstrip()
            fingerprint = text.casefold()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            # ``value`` is submitted as the person's next message and ``label``
            # is what the chip reads; the client falls back to the value when a
            # label is missing, so sending both keeps that path unexercised.
            buttons.append({"value": text, "label": text})
            if len(buttons) >= limit:
                break
        return buttons
