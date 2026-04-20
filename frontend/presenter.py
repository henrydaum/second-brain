from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from frontend.types import FrontendAction, PlatformCapabilities


class FrontendPresenter:
    def tool_started(self, tool_name: str, caps: PlatformCapabilities) -> FrontendAction:
        return FrontendAction(
            type="show_status",
            text=f"\u23f3 {tool_name}",
            status_id=uuid4().hex if caps.supports_message_edit else None,
        )

    def tool_finished(self, tool_name: str, result, caps: PlatformCapabilities,
                      status_id: str | None = None) -> list[FrontendAction]:
        icon = "\u2705" if result.success else "\u274c"
        actions = [FrontendAction(
            type="update_status" if status_id and caps.supports_message_edit else "show_status",
            text=f"{icon} {tool_name}",
            status_id=status_id,
        )]
        if tool_name == "render_files" and result.attachment_paths:
            actions.append(FrontendAction(
                type="send_attachments",
                attachments=list(result.attachment_paths),
                text=result.llm_summary or "",
            ))
        return actions

    def approval_request(self, req, caps: PlatformCapabilities) -> FrontendAction:
        buttons = []
        if caps.supports_buttons:
            buttons = [
                {"label": "\u274c Deny", "value": "deny"},
                {"label": "\u2705 Allow", "value": "allow"},
            ]
        footer = "" if buttons else "\n\nRespond with /allow or /deny."
        return FrontendAction(
            type="show_choices",
            text=f"Agent requests approval:\n{req.command}\n\n{req.reason}{footer}",
            buttons=buttons,
            metadata={"request_id": req.id, "choice_prefix": "approval"},
        )

    def approval_resolved(self, req, approved: bool, resolved_by: str | None,
                          adapter_name: str) -> FrontendAction:
        verdict = "\u2705 Allowed" if approved else "\u274c Denied"
        if resolved_by and resolved_by != adapter_name:
            note = f"{verdict} (resolved via {resolved_by})"
        else:
            note = verdict
        return FrontendAction(
            type="resolve_choices",
            text=note,
            metadata={"request_id": req.id, "approved": approved, "resolved_by": resolved_by or ""},
        )

    def pushed_message(self, payload: dict, caps: PlatformCapabilities) -> FrontendAction:
        title = str(payload.get("title") or "").strip()
        kind = str(payload.get("kind") or "").strip()
        message = str(payload.get("message") or "").strip()
        lines = []
        if title:
            lines.append(f"**{title}**" if caps.supports_rich_text else title)
        elif kind:
            label = kind.title()
            lines.append(f"**{label}**" if caps.supports_rich_text else label)
        if message:
            lines.append(message)
        return FrontendAction(type="send_message", text="\n\n".join(lines).strip())

    def notice(self, text: str) -> FrontendAction:
        return FrontendAction(type="send_message", text=f"[{text}]")

    def choice_menu(self, title: str, choices: list[dict], caps: PlatformCapabilities,
                    choice_prefix: str, footer: str = "") -> FrontendAction:
        if caps.supports_buttons:
            return FrontendAction(
                type="show_choices",
                text=title,
                buttons=[{"label": c["label"], "value": c["value"]} for c in choices],
                metadata={"choice_prefix": choice_prefix},
            )
        lines = [title]
        for choice in choices:
            lines.append(f"  {choice['label']}")
        if footer:
            lines.append("")
            lines.append(footer)
        return FrontendAction(type="show_choices", text="\n".join(lines))

    def history_menu(self, conversations: list[dict], caps: PlatformCapabilities) -> FrontendAction:
        choices = []
        for conv in conversations:
            title = (conv["title"] or "New conversation").replace("\n", " ")[:40]
            ts = conv.get("updated_at")
            time_str = datetime.fromtimestamp(ts).strftime("%b %d") if ts else ""
            label = f"{title}  ({time_str})" if time_str else title
            choices.append({"label": label, "value": str(conv["id"])})
        return self.choice_menu("Recent conversations:", choices, caps, "hist")

    def form_field(self, name: str, field_type: str = "string", description: str = "",
                   required: bool = False, enum: list | None = None,
                   choice_prefix: str | None = None, choice_context: str | None = None) -> FrontendAction:
        req = " (required)" if required else " (optional, send /skip)"
        desc = f"\n{description}" if description else ""
        if field_type == "string":
            hint = "\nType your value as plain text (no quotes needed)."
        elif field_type == "integer":
            hint = "\nType a whole number, e.g. `42`."
        elif field_type == "number":
            hint = "\nType a number, e.g. `3.14`."
        elif field_type == "array":
            hint = "\nSend each item on its own line."
        elif field_type == "object":
            hint = "\nSend as JSON, e.g. `{\"key\": \"value\"}`."
        else:
            hint = ""
        buttons = []
        metadata = {}
        if choice_prefix:
            metadata["choice_prefix"] = choice_prefix
        if choice_context is not None:
            metadata["choice_context"] = choice_context
        if enum:
            buttons = [{"label": str(v), "value": str(v)} for v in enum]
        elif field_type == "boolean":
            buttons = [{"label": "True", "value": "true"}, {"label": "False", "value": "false"}]
        return FrontendAction(
            type="request_form_input",
            text=f"{name} ({field_type}){req}{desc}{hint}",
            buttons=buttons,
            metadata=metadata,
        )
