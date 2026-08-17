"""OpenAI backend for a ChatGPT *subscription*, not a platform API key.

The installed LiteLLM backend already reaches ``openai/*``, and for an API key
that is the right file — this one would be redundant. What LiteLLM cannot do
is authenticate with a subscription, because its only credential input is a
key string and a subscription is an OAuth token pair with a lifetime.

So the split is: ``service_openai_auth`` owns the token pair and renews it,
and this file places calls with whatever that service currently holds. Neither
half is useful alone.

NO PROVIDER LIBRARY, ON PURPOSE
-------------------------------
This talks to the endpoint with ``sdk.net.http``, not the ``openai`` package,
and the template says why better than a summary would: *"If you can reach the
provider over plain HTTP, prefer sdk.net.http and the handle — you get the
substitution back."*

That is the whole design. The access token arrives here as a
``<secret:...>`` handle and is put straight into a header; the kernel swaps in
the real value on the way out. This box never holds the credential, so it
cannot log it, cannot put it in an error message, and cannot leak it into a
traceback. It also stays stdlib-only, which means no pip dependency and a box
that starts instantly.

WHAT IT COSTS: NO STREAMING
---------------------------
``net.http`` reads the whole response before answering — the Request is a
round trip, not a pipe — so there is no way to emit ``sdk.llm.delta`` while
the model is still talking. ``supports_streaming`` is therefore False and the
kernel simply never sets ``request.stream``.

This is a real trade and worth stating plainly rather than burying: streaming
is available only by using a provider library that opens its own socket, and
that library would need the token in plaintext, inside this box, to use it.
Handle-safety and live tokens are mutually exclusive here. The choice made is
safety, because a subscription credential is worth more than a typing
animation — but it is a choice, and reversing it means importing an HTTP
library and calling ``sdk.secrets.reveal``.

ISOLATION
---------
With no foreign import, ``sandbox/isolation.py`` resolves this file IN_PROCESS
in the installed tree. That is correct here and would not be for a normal
backend: the risk the ``llm/`` migration existed to contain was a volatile
provider SDK plus an unmediated socket, and this file has neither. Every
effect it performs is a Request the kernel classifies. Same reasoning
``parse_text`` gets for staying in-process.
"""

# ── DECLARATIONS ────────────────────────────────────────────────────────
#
# Read by AST, never by importing this file.

lifetime = "persistent"

# The auth service travels with this file. Neither half is useful alone: this
# backend has no way to authenticate without it, so installing one and not the
# other leaves a profile that fails every call.
#
# Two things this is *not*. It is not an import — the service is reached
# through ``sdk.services.call``, so isolation is unaffected (the resolver
# intersects what is declared here with what the AST actually imports, and
# that intersection is empty). And it is not ownership: uninstall walks this
# edge *backwards*, so removing ``service_openai_auth`` correctly takes this
# backend with it, while removing this backend leaves the service in place.
# Same relationship ``parsers/parse_gdoc.py`` declares with ``service_drive``.
dependencies_files = ["services/service_openai_auth.py"]

# See "WHAT IT COSTS" above. Not a limitation of the provider — a consequence
# of refusing to hold the credential.
supports_streaming = False

supports_tool_choice = True

# Images only. Audio and video would need this file to know each endpoint's
# encoding for them, and claiming a modality it cannot actually put on the
# wire would route attachments here that should have gone to the text
# fallback — a failure the user sees as a model ignoring their file.
native_modalities = ["image"]

display_name = "OpenAI (ChatGPT subscription)"

import base64

from guest.llm import BaseLLMBackend, LLMResponse

# Image modality to a media type, for the data URLs the wire format wants.
# Keyed off the file name because that is what the kernel routes with.
_IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


class OpenAISubscriptionBackend(BaseLLMBackend):
    """Reach a ChatGPT subscription endpoint over mediated HTTP."""

    def start(self, sdk):
        """Nothing to acquire.

        There is no library to import and no client to build — every call is
        an ``sdk.net.http``, and the connection settings arrive per request so
        that two boxes running this file stay interchangeable and the pool
        works. Holding anything profile-shaped here is the one mistake this
        contract really punishes.
        """
        return True

    def stop(self, sdk):
        """Nothing was opened."""
        return True

    # ── the call ────────────────────────────────────────────────────

    def chat(self, sdk, request):
        """Place one call and answer with what the model said."""
        grant = sdk.services.call("openai_auth", "token") or {}
        if not grant.get("authorized"):
            # Fail rather than wait. What an unauthorized grant is waiting for
            # is a person finishing a sign-in on their phone, and a turn that
            # blocks on that is a frozen conversation — the notification the
            # auth service raised is the thing that will actually resolve it.
            return LLMResponse.failure(
                f"OpenAI subscription is not signed in "
                f"({grant.get('detail') or 'no token'}).",
                "not_authorized")

        url = request.base_url or grant.get("url") or ""
        if not url:
            return LLMResponse.failure(
                "no OpenAI endpoint configured; set openai_responses_url in "
                "/config or an endpoint on the profile", "not_configured")

        answer = sdk.net.http_json(
            url, method="POST",
            headers={
                # A handle, not a token. Substituted on the way out.
                "Authorization": f"Bearer {grant['token']}",
                "Content-Type": "application/json",
            },
            json=self._payload(sdk, request),
        )
        return self._read(answer)

    # ── the wire ────────────────────────────────────────────────────
    #
    # The two methods below are the only part of this file that knows the
    # endpoint's shape. They are written against the chat-completions schema,
    # which is what ``LLMRequest.messages`` already *is* — the kernel builds
    # role/content dicts with ``tool_calls`` and ``tool_call_id``, so the
    # mapping is close to identity. If the subscription endpoint speaks a
    # different schema, this pair is what changes and nothing else does.

    def _payload(self, sdk, request):
        """Build the request body."""
        payload = {
            "model": request.model_name,
            "messages": self._with_attachments(sdk, request),
        }
        if request.tools:
            payload["tools"] = request.tools
        # Forwarded as-is, which is what carries ``tool_choice`` when a
        # doorway forces one, plus whatever the profile set.
        payload.update(request.params or {})
        return payload

    def _read(self, answer):
        """Turn one HTTP answer into an ``LLMResponse``.

        An error *status* is data here rather than an exception, because
        ``net.http`` hands back the body of a 4xx — which is where an API
        explains itself. Raising on the status would discard the explanation
        and leave the kernel guessing.
        """
        status = answer.get("status", 0)
        body = answer.get("body")
        if not isinstance(body, dict):
            return LLMResponse.failure(
                f"OpenAI returned {status} with an unreadable body")

        if status >= 400 or "error" in body:
            detail = body.get("error")
            message = (detail.get("message") if isinstance(detail, dict)
                       else detail) or f"OpenAI returned {status}"
            # Left for ``__chat__`` and ``LLMResponse.is_context_limit_error``
            # to classify from the text: every provider spells an overflow
            # differently and ``guest.llm`` already owns that heuristic. A
            # context error has to be *recognised* rather than guessed at
            # downstream, because it is what triggers compaction and a retry
            # instead of a failed turn.
            return LLMResponse.failure(str(message))

        choices = body.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=[
                {
                    "id": call.get("id") or f"call_{index}",
                    "name": (call.get("function") or {}).get("name") or "",
                    # A JSON *string*, which is the contract — the kernel
                    # parses it. Defaulted rather than left empty because a
                    # tool call with no arguments still has to decode.
                    "arguments": ((call.get("function") or {})
                                  .get("arguments") or "{}"),
                }
                for index, call in enumerate(message.get("tool_calls") or [])
                if (call.get("function") or {}).get("name")
            ],
            prompt_tokens=usage.get("prompt_tokens"),
            cached_prompt_tokens=details.get("cached_tokens"),
        )

    def _with_attachments(self, sdk, request):
        """Inline the media the kernel already decided this model can read.

        Routing happened kernel-side: anything the model cannot ingest
        natively was turned into text and appended to the last user message
        before this file saw it. So everything in ``request.attachments`` is
        meant to go on the wire, and there is no capability check to repeat.

        Bytes come from ``sdk.fs.read_bytes``, never ``open`` — and never
        ``sdk.fs.read``, which decodes as UTF-8 with replacement and would
        hand back a mangled JPEG.
        """
        if not request.attachments:
            return request.messages

        blocks = []
        for item in request.attachments:
            name = (item.get("file_name") or "").lower()
            suffix = name[name.rfind("."):] if "." in name else ""
            media = _IMAGE_TYPES.get(suffix, "image/png")
            raw = sdk.fs.read_bytes(item["path"])
            # read_bytes answers base64 already; tolerate raw bytes in case
            # that ever changes rather than silently sending "b'\\x89PNG...'".
            data = raw if isinstance(raw, str) else base64.b64encode(
                raw).decode("ascii")
            blocks.append({"type": "image_url",
                           "image_url": {"url": f"data:{media};base64,{data}"}})

        messages = [dict(message) for message in request.messages]
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                content = messages[index].get("content")
                messages[index]["content"] = [
                    {"type": "text", "text": str(content or "")}, *blocks]
                break
        return messages
