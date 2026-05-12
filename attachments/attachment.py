"""Attachment support for attachment."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


@dataclass
class Attachment:
    """Attachment."""
    path: str
    extension: str
    file_name: str
    modality: str  # "image" | "audio" | "video" | "text" | "binary" | ...
    parsed_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Handle to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attachment":
        """Handle from dict."""
        fields = cls.__dataclass_fields__
        return cls(**{k: data.get(k) for k in fields if k in data})


@dataclass
class AttachmentBundle:
    """Attachment bundle."""
    items: list[Attachment] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Internal helper to handle bool."""
        return bool(self.items)

    def __iter__(self):
        """Internal helper to handle iter."""
        return iter(self.items)

    def __len__(self) -> int:
        """Internal helper to handle len."""
        return len(self.items)

    def append(self, attachment: Attachment) -> None:
        """Handle append."""
        self.items.append(attachment)

    def for_llm(self, capabilities: dict[str, bool | None] | None) -> tuple[list[str], str]:
        """Route each attachment by the LLM's capabilities dict.

        Returns ``(native_paths, suffix_text)``:
        - ``native_paths`` lists files the LLM can ingest directly (e.g.
          images for a vision model). Caller inlines these via its
          existing native-image path.
        - ``suffix_text`` is appended to the last user message and carries
          parsed-text blurbs for non-native files plus pointer-fallback
          lines for files we couldn't parse.
        """
        caps = capabilities or {}
        native_paths: list[str] = []
        suffix_parts: list[str] = []
        for att in self.items:
            if caps.get(att.modality):
                native_paths.append(att.path)
                continue
            if att.parsed_text:
                blurb = (
                    f"The user attached a {att.modality} file ({att.file_name}). "
                    f"Parsed contents:\n{att.parsed_text}"
                )
            else:
                blurb = (
                    f"The user attached a file: {att.file_name}. "
                    f"It has been saved into {att.path}."
                )
            suffix_parts.append(blurb)
        return native_paths, "\n\n".join(suffix_parts)

    def to_list(self) -> list[dict[str, Any]]:
        """Handle to list."""
        return [a.to_dict() for a in self.items]

    @classmethod
    def from_iterable(cls, data: Iterable[Any] | None) -> "AttachmentBundle":
        """Handle from iterable."""
        if not data:
            return cls()
        if isinstance(data, AttachmentBundle):
            return data
        items: list[Attachment] = []
        for entry in data:
            if isinstance(entry, Attachment):
                items.append(entry)
            elif isinstance(entry, dict):
                items.append(Attachment.from_dict(entry))
        return cls(items)
