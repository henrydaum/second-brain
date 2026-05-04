from events.ui_request import InteractiveRequest


class ApprovalRequest(InteractiveRequest):
    """
    Object-oriented tracking for agent approval requests.

    Replaces the raw dictionary payload on the event bus, giving frontends
    a clean API to interact with standard thread synchronization primitives
    without needing to understand them.
    """
    kind = "approval_request"

    def __init__(self, command: str, reason: str):
        super().__init__(
            title="Agent requests approval",
            body=reason,
            metadata={"command": command, "reason": reason},
        )
        self.command = command
        self.reason = reason
        self.approved = False

    def resolve(self, approved: bool):
        """
        Mark this request as handled. Can only be called once.
        Automatically broadcasts an APPROVAL_RESOLVED event to notify other
        frontends that they can clean up their UI placeholders.
        """
        self.approved = approved
        super().resolve(approved)

    def on_resolved(self):
        # Avoid circular imports at module load time by importing bus here
        from events.event_bus import bus
        from events.event_channels import APPROVAL_RESOLVED
        bus.emit(APPROVAL_RESOLVED, self)
