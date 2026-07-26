"""A helper in the wordcount box.

No base class, no declaration beyond its box. Helpers are ordinary modules —
the only thing that makes this one special is that it shares an execution
context with the tool that imports it.
"""

box = "wordcount"


def count_words(text: str) -> int:
    """Count whitespace-separated words."""
    return len((text or "").split())
