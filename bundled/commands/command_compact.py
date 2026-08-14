"""Slash command plugin for `/compact`."""

from guest.bases import BaseCommand


class CompactCommand(BaseCommand):
    """Slash-command handler for `/compact`.

    Kernel rather than store, for the reason `/mode` and `/permissions` are:
    it operates one kernel-owned mechanism, and a context-safety lever that
    stops existing when a package is uninstalled is worse than none.
    """
    name = "compact"
    description = "Summarize the conversation so far and shrink the context"
    category = "Conversation"
    # Asked up front, which is the right path for a command: the grant is
    # stated and answered before the body runs, rather than interrupting a
    # half-done run with one Request in isolation. ``session.compact`` is
    # consequential because nothing removes a compaction marker — the
    # conversation has no way back to being read in full — so the state
    # machine asks even though the person typed this themselves.
    require_approval = True
    approval_actor_id = "user"
    requests = ["ui.progress", "session.compact"]

    def run(self, sdk, args):
        """Execute `/compact` for the active session."""
        # The compactor places a real model call, so this is worth narrating on
        # the line the person is already watching. Not ``session.push`` — that
        # destination is the conversation, which a command never speaks into.
        sdk.ui.progress("Compacting conversation...")
        try:
            report = sdk.session.compact()
        except sdk.Failed as exc:
            return f"Could not compact: {exc.error}"
        saved = report.get("chars_saved") or 0
        # ASCII arrows: this lands on a Windows console under cp1252, where a
        # unicode arrow raises rather than renders.
        return sdk.md.card("Compacted", [
            ("Messages", f"{report.get('messages_before', 0):,} -> "
                         f"{report.get('messages_after', 0):,}"),
            ("Characters", f"{report.get('chars_before', 0):,} -> "
                           f"{report.get('chars_after', 0):,}"),
            ("Saved", f"{saved:,} chars ({self._percent(report)})"),
            ("Summary", f"{report.get('summary_chars', 0):,} chars"),
        ])

    @staticmethod
    def _percent(report) -> str:
        """How much smaller, as a share of what was there.

        The absolute figure alone does not say whether it was worth doing:
        40,000 characters saved means something different out of 45,000 than
        out of 400,000.
        """
        before = report.get("chars_before") or 0
        if not before:
            return "0%"
        return f"{(report.get('chars_saved') or 0) / before:.0%}"
