"""ChatGPT-plan Codex backend using the Responses API."""



dependencies_files = ['services/service_codex_auth.py']
dependencies_pip = ['Pillow', 'openai>=2.30.0']

import base64
import io
import json
import mimetypes

import openai

from guest.llm import BaseLLMBackend, LLMResponse


lifetime = "persistent"
timeout = 600
supports_streaming = True
supports_tool_choice = False
# The registry's empty declaration means "use the legacy all-modalities
# default". A non-media sentinel therefore spells text-only honestly.
native_modalities = ["image", "audio"]
display_name = "Codex (ChatGPT plan)"

BASE_URL = "https://chatgpt.com/backend-api/codex"
MODELS = [
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
    "gpt-5.4-mini", "gpt-5.4", "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
]
IMAGE_EDGE = 2048
IMAGE_QUALITY = 88
MAX_IMAGE_PIXELS = 100_000_000


class CodexBackend(BaseLLMBackend):
    """Call the private Codex Responses endpoint with service-owned OAuth."""

    def start(self, sdk):
        self._openai = openai
        return True

    def stop(self, sdk):
        self._openai = None
        return True

    def chat(self, sdk, request):
        token = sdk.services.call("codex_auth", "access_token")
        payload = self._payload(request)
        self._attach_media(sdk, request, payload)
        headers = self._headers(token)
        pieces = []
        calls = {}
        usage = {}
        client = self._openai.OpenAI(
            api_key=token,
            base_url=(request.base_url or BASE_URL).rstrip("/"),
            default_headers=headers,
            timeout=600.0,
        )
        try:
            for event in client.responses.create(**payload):
                data = event.model_dump() if hasattr(event, "model_dump") else event
                if isinstance(data, dict):
                    self._accept(sdk, request, data, pieces, calls, usage)
        except Exception as exc:
            raise RuntimeError(f"Codex request failed: {exc}")
        ordered = []
        for key in sorted(calls, key=str):
            call = calls[key]
            if call.get("name"):
                ordered.append({
                    "id": call.get("id") or f"call_{key}",
                    "name": call["name"],
                    "arguments": call.get("arguments") or "{}",
                })
        return LLMResponse(
            content="".join(pieces),
            tool_calls=ordered,
            prompt_tokens=usage.get("input_tokens"),
            cached_prompt_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )

    def providers(self, sdk, provider=""):
        if provider and provider != "codex":
            return []
        return [{
            "id": "codex",
            "label": "OpenAI Codex",
            "endpoint": BASE_URL,
            "description": "Uses a ChatGPT subscription authenticated by /codex.",
        }]

    def models(self, sdk, endpoint, api_key, provider="", live=False):
        try:
            catalog = sdk.services.call("codex_auth", "model_catalog") or []
            rows = [{
                "name": row["slug"],
                "label": row.get("display_name") or row["slug"],
                "description": row.get("description") or "",
            } for row in catalog
                    if isinstance(row, dict) and row.get("slug")]
            if not rows:
                names = sdk.services.call("codex_auth", "models") or []
                rows = [{"name": name, "label": name} for name in names
                        if isinstance(name, str) and name]
            if rows:
                return rows
        except Exception as exc:
            sdk.log(f"Codex model cache unavailable: {exc}", level="debug")
        return [{"name": name, "label": name} for name in MODELS]

    def info(self, sdk, model_name, endpoint=""):
        cached = self._model_info(sdk, model_name)
        if cached:
            row = {"description": cached.get("description") or (
                "A Codex model available to this ChatGPT account.")}
            window = cached.get("context_window")
            if isinstance(window, (int, float)) and window > 0:
                row["context_size"] = int(window)
            return [row]
        row = {
            "description": (
                "A Codex model available to this ChatGPT account. OpenAI's "
                "allow-list changes, so the model ID is entered explicitly."
            )
        }
        name = (model_name or "").lower()
        if name == "gpt-5.3-codex-spark":
            row["context_size"] = 128000
        elif name.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.6")):
            row["context_size"] = 272000
        return [row]

    def params(self, sdk, model_name, endpoint):
        cached = self._model_info(sdk, model_name)
        if cached:
            out = []
            efforts = cached.get("supported_reasoning_levels") or []
            if efforts:
                default = cached.get("default_reasoning_level")
                note = f"Model default: {default}." if default else ""
                out.append({
                    "name": "reasoning_effort", "label": "Reasoning effort",
                    "kind": "choice", "choices": efforts,
                    "supported": True, "note": note,
                    "description": "Controls how much reasoning the model performs.",
                })
            if cached.get("support_verbosity"):
                default = cached.get("default_verbosity")
                note = f"Model default: {default}." if default else ""
                out.append({
                    "name": "verbosity", "label": "Verbosity",
                    "kind": "choice", "choices": ["low", "medium", "high"],
                    "supported": True, "note": note,
                    "description": "Controls the detail of text responses.",
                })
            return out
        return [{
            "name": "reasoning_effort", "label": "Reasoning effort",
            "kind": "choice",
            "choices": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            "supported": True,
            "note": "Live metadata unavailable; unsupported values may be rejected.",
            "description": "Controls how much reasoning the model performs.",
        }]

    def _model_info(self, sdk, model_name):
        try:
            rows = sdk.services.call("codex_auth", "model_catalog") or []
        except Exception as exc:
            sdk.log(f"Codex model metadata unavailable: {exc}", level="debug")
            return {}
        wanted = (model_name or "").lower()
        for row in rows:
            if isinstance(row, dict) and str(row.get("slug") or "").lower() == wanted:
                return row
        return {}

    def _payload(self, request):
        instructions = []
        input_items = []
        for message in request.messages:
            role = message.get("role") or "user"
            content = message.get("content") or ""
            if role in ("system", "developer"):
                instructions.append(self._text(content))
                continue
            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": self._text(content),
                })
                continue
            if role == "assistant" and message.get("tool_calls"):
                text = self._text(content)
                if text:
                    input_items.append({"role": "assistant", "content": text})
                for call in message.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": fn.get("name") or call.get("name") or "",
                        "arguments": fn.get("arguments") or call.get("arguments") or "{}",
                    })
                continue
            input_items.append({"role": role, "content": self._text(content)})
        payload = {
            "model": request.model_name,
            "instructions": "\n\n".join(part for part in instructions if part),
            "input": input_items,
            "stream": True,
            "store": False,
        }
        tools = []
        for tool in request.tools or []:
            fn = tool.get("function") if tool.get("type") == "function" else tool
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            tools.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                "strict": False,
            })
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        params = dict(request.params or {})
        effort = params.pop("reasoning_effort", None)
        if effort is not None:
            payload["reasoning"] = {"effort": effort, "summary": "auto"}
            payload["include"] = ["reasoning.encrypted_content"]
        verbosity = params.pop("verbosity", None)
        if verbosity is not None:
            payload["text"] = {"verbosity": verbosity}
        payload.update(params)
        payload["stream"] = True
        payload["store"] = False
        return payload

    def _headers(self, token):
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "OpenAI-Beta": "responses=experimental",
            "User-Agent": "SecondBrain/1",
            "originator": "second-brain",
        }
        try:
            part = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
            account = (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
            if account:
                headers["ChatGPT-Account-ID"] = account
        except Exception:
            pass
        return headers

    def _attach_media(self, sdk, request, payload):
        """Append native image/audio blocks to the last user message."""
        blocks = []
        fallbacks = []
        for item in request.attachments or []:
            try:
                block = self._media_block(sdk, item)
            except Exception as exc:
                sdk.log(
                    f"Could not inline {item.get('file_name')}: {exc}",
                    level="warning")
                block = None
            if block:
                blocks.append(block)
            else:
                fallbacks.append(self._attachment_fallback(item))
        if not blocks and not fallbacks:
            return
        for message in reversed(payload.get("input") or []):
            if message.get("role") != "user":
                continue
            content = message.get("content") or ""
            parts = content if isinstance(content, list) else [
                {"type": "input_text", "text": str(content)}]
            if fallbacks:
                parts.append({
                    "type": "input_text", "text": "\n\n".join(fallbacks)})
            message["content"] = [*parts, *blocks]
            return

    def _media_block(self, sdk, item):
        path = item.get("path") or ""
        modality = item.get("modality")
        if modality == "image":
            return {
                "type": "input_image",
                "image_url": self._image_data_url(sdk, path),
                "detail": "auto",
            }
        if modality == "audio":
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if not mime.startswith("audio/"):
                return None
            data = base64.b64encode(sdk.fs.read_bytes(path)).decode("ascii")
            return {
                "type": "input_audio",
                "audio_url": f"data:{mime};base64,{data}",
            }
        return None

    @staticmethod
    def _attachment_fallback(item):
        parsed = str(item.get("parsed_text") or "").strip()
        name = item.get("file_name") or "attachment"
        if parsed:
            return f"The user attached {name}. Parsed contents:\n{parsed}"
        return f"The user attached {name}, but it could not be sent natively."

    def _image_data_url(self, sdk, path):
        from PIL import Image, ImageFile

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = None
        try:
            image = Image.open(io.BytesIO(sdk.fs.read_bytes(path)))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((IMAGE_EDGE, IMAGE_EDGE), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(
                buffer, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{data}"
        finally:
            if image is not None:
                image.close()

    def _accept(self, sdk, request, event, pieces, calls, usage):
        kind = event.get("type") or ""
        if kind == "response.output_text.delta":
            delta = event.get("delta") or ""
            if delta:
                pieces.append(delta)
                if request.stream:
                    sdk.llm.delta(delta)
        elif kind == "response.function_call_arguments.delta":
            key = event.get("item_id") or event.get("output_index", len(calls))
            entry = calls.setdefault(key, {"id": None, "name": None, "arguments": ""})
            entry["arguments"] += event.get("delta") or ""
        elif kind in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                key = item.get("id") or event.get("output_index", len(calls))
                entry = calls.setdefault(key, {"id": None, "name": None, "arguments": ""})
                entry["id"] = item.get("call_id") or entry["id"]
                entry["name"] = item.get("name") or entry["name"]
                if item.get("arguments"):
                    entry["arguments"] = item["arguments"]
        elif kind in ("response.completed", "response.done"):
            response = event.get("response") or {}
            if isinstance(response.get("usage"), dict):
                usage.update(response["usage"])
        elif kind in ("response.failed", "error"):
            error = event.get("error") or (event.get("response") or {}).get("error") or event
            raise RuntimeError(f"Codex stream failed: {json.dumps(error, default=str)[:2000]}")

    @staticmethod
    def _text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content or "")
