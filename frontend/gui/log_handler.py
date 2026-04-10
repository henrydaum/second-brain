"""
Thread-safe logging handler that captures records into a bounded deque for GUI display.
"""

import collections
import logging


class GuiLogHandler(logging.Handler):
    """Thread-safe handler that stores formatted records and notifies the GUI."""

    def __init__(self, max_records=500):
        """Initialize with a bounded deque of *max_records* and no callback."""
        super().__init__()
        self._records = collections.deque(maxlen=max_records)
        self._on_record = None

    def set_callback(self, fn):
        """Register *fn(formatted_str, record)* to be invoked on each emit."""
        self._on_record = fn

    @property
    def records(self):
        return list(self._records)

    def emit(self, record):
        """Store the formatted record and notify the callback if set."""
        formatted = self.format(record)
        self._records.append((formatted, record))
        if self._on_record:
            try:
                self._on_record(formatted, record)
            except Exception:
                pass
