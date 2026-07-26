"""A tool written against the new contracts.

Imports ``guest`` the same way in both runners: the child has ``sandbox/`` as
its working directory, and in-process the package is aliased.
"""

from guest.bases import BaseTool

from .helper_words import count_words


class WordCount(BaseTool):
    """Count the words in a text file."""

    name = "word_count"
    description = "Count the words in a text file."
    box = "wordcount"
    requests = ["fs.read"]
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def run(self, sdk, path):
        """Read the file and count."""
        return count_words(sdk.fs.read(path))
