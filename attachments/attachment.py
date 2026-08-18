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

    def record(self) -> dict[str, Any]:
        """The durable half: what the file *is*, not what it said.

        This is what a transcript row keeps. ``parsed_text`` deliberately does
        not travel — it is a rendering of the file for one model on one turn,
        it can be four thousand characters of a PDF, and the file itself is
        still on disk to be parsed again. ``metadata`` goes with it for the
        same reason.
        """
        return {"path": self.path, "file_name": self.file_name,
                "modality": self.modality, "extension": self.extension}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attachment":
        """Handle from dict."""
        fields = cls.__dataclass_fields__
        return cls(**{k: data.get(k) for k in fields if k in data})


def pointer_for(records: Iterable[dict[str, Any]] | None) -> str:
    """The line that tells a model a file arrived, one per record.

    A transcript row stores the person's own words and the files as *records*;
    this is how the two become one message again at prompt time. It has to stay
    exactly what ``parse_attachment`` used to weld into the text, because every
    conversation written before the column existed still has these lines in its
    content and the model must not see two spellings of one thing.
    """
    return "\n".join(
        f"[Attached {record.get('modality') or 'binary'} file: "
        f"{record.get('file_name') or 'attachment'} "
        f"(cached at {record.get('path') or ''})]"
        for record in records or [])


def with_pointers(content: str, records: Iterable[dict[str, Any]] | None) -> str:
    """One message's text as a model should read it: words, then files."""
    pointer = pointer_for(records)
    if not pointer:
        return content or ""
    return f"{content}\n\n{pointer}" if content else pointer


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

    def split_for_llm(
        self,
        capabilities: dict[str, bool | None] | None,
        native_modalities: set[str] | frozenset[str] | None = None,
    ) -> tuple["AttachmentBundle", str]:
        """Route each attachment by the LLM's capabilities dict.

        Returns ``(native_bundle, suffix_text)``:
        - ``native_bundle`` carries files the model and backend can ingest
          directly. Backend plugins serialize it into their own wire format.
        - ``suffix_text`` is appended to the last user message and carries
          parsed-text blurbs for non-native files plus pointer-fallback
          lines for files we couldn't parse.
        """
        caps = capabilities or {}
        native_modalities = native_modalities or set()
        native = AttachmentBundle()
        suffix_parts: list[str] = []
        for att in self.items:
            if caps.get(att.modality) and att.modality in native_modalities:
                native.append(att)
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
        return native, "\n\n".join(suffix_parts)

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
