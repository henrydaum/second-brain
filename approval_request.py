import threading
import uuid

class ApprovalRequest:
    """
    Object-oriented tracking for agent approval requests.
    
    Replaces the raw dictionary payload on the event bus, giving frontends
    a clean API to interact with standard thread synchronization primitives
    without needing to understand them.
    """
    def __init__(self, command: str, reason: str):
        self.id = uuid.uuid4().hex
        self.command = command
        self.reason = reason
        self.approved = False
        self._event = threading.Event()

    @property
    def is_resolved(self) -> bool:
        """Returns True if this request has already been handled by a frontend."""
        return self._event.is_set()

    def resolve(self, approved: bool):
        """
        Mark this request as handled. Can only be called once.
        Automatically broadcasts an APPROVAL_RESOLVED event to notify other
        frontends that they can clean up their UI placeholders.
        """
        if self._event.is_set():
            return
        
        self.approved = approved
        self._event.set()
        
        # Avoid circular imports at module load time by importing bus here
        from event_bus import bus
        from event_channels import APPROVAL_RESOLVED
        bus.emit(APPROVAL_RESOLVED, self)

    def wait(self, timeout: float = 300.0) -> bool:
        """
        Block the current thread until another thread calls .resolve().
        Returns True if resolved in time, False if it timed out.
        """
        return self._event.wait(timeout=timeout)
