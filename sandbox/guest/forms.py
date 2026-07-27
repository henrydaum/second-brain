"""Pure, serializable form values for sandboxed commands.

The kernel owns form progression, coercion, and validation.  A guest command
only describes the fields it wants collected, so a form step is deliberately
just a dictionary with a convenient, typed constructor.  Being a real mapping
keeps it JSON-safe across subprocess boundaries without a custom codec.

Callable validators are intentionally absent: executable objects cannot cross
the sandbox boundary.  The kernel's reconstructed FormStep applies its normal
type, enum, and path validation after this data arrives.
"""

from __future__ import annotations

from typing import Any


class FormStep(dict):
    """One field in a dependent command form."""

    def __init__(
        self,
        name: str,
        prompt: str = "",
        required: bool = True,
        type: str = "string",
        enum: list[Any] | None = None,
        enum_labels: list[str] | None = None,
        default: Any = None,
        prompt_when_missing: bool = False,
        columns: int | None = None,
    ):
        super().__init__(
            name=name,
            prompt=prompt,
            required=required,
            type=type,
            enum=enum,
            enum_labels=enum_labels,
            default=default,
            prompt_when_missing=prompt_when_missing,
            columns=columns,
        )

    def to_dict(self) -> dict:
        """Return a detached plain-dictionary representation."""
        return dict(self)
