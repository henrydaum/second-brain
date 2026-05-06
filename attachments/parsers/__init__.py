"""Importing this package triggers ``register(...)`` calls in each
``parser_*.py`` module so the attachment registry is populated.

To add a new parser: drop a new ``parser_*.py`` file in this directory
that calls ``attachments.registry.register(...)`` at module bottom, then
add a matching ``from . import parser_<name>`` line below.
"""

from attachments.parsers import parser_text  # noqa: F401
from attachments.parsers import parser_pdf   # noqa: F401
from attachments.parsers import parser_audio  # noqa: F401
