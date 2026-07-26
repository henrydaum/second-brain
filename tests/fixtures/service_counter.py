"""A resident service: state acquired once, kept, called into repeatedly."""

from guest.bases import BaseService


class Counter(BaseService):
    """Holds a running total across calls."""

    name = "counter"
    description = "Accumulates numbers across calls."
    box = "counter"
    exports = ["add", "total", "read_file"]

    def start(self, sdk):
        """Acquire the state this service holds."""
        self.running = 0
        self.started = True
        sdk.log("counter service started")
        return True

    def stop(self, sdk):
        """Release it."""
        self.started = False
        return True

    def add(self, sdk, n):
        """Add to the total and return the new value."""
        self.running += n
        return self.running

    def total(self, sdk):
        """Report the total without changing it."""
        return self.running

    def read_file(self, sdk, path):
        """A method that makes a Request mid-call."""
        return sdk.fs.read(path)

    def explode(self, sdk):
        """Fail, to prove one bad call does not kill the service."""
        raise ValueError("bad call")

    def hang(self, sdk):
        """Never finish, to exercise the per-call deadline."""
        while True:
            sdk.fs.list(".")
