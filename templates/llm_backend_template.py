"""LLM BACKEND TEMPLATE — how Second Brain talks to a model.

Copy this to ``helpers/llm_<provider>.py`` in whichever plugin tree you are
working in. The name matters: ``llm_*.py`` under ``helpers/`` at a tree's root
is how the registry finds you.

A backend is **not a plugin**. There is no family, no entry point, and nothing
discovery registers — same as a parser. It is a class the kernel loads into a
box and calls, and it lives under ``helpers/`` precisely because it belongs to
no family: it is a way of reaching a model, not a capability the agent has.

WHAT YOU ARE HANDED, AND WHAT YOU GIVE BACK
    def chat(self, sdk, request: LLMRequest) -> LLMResponse

That is the whole contract. ``request`` carries the model name, the messages,
the tool schemas, the connection settings, and any attachments the kernel has
already decided this model can read natively. You return one ``LLMResponse``,
whether you streamed or not.

WHY THE PROFILE IS ON THE REQUEST AND NOT ON YOU
Every connection detail arrives per call, so two boxes running this file are
interchangeable. That is what lets the kernel keep a *pool* of them and serve
concurrent calls in parallel — a scheduled subagent does not queue behind a
foreground turn. Storing ``self.model_name`` in ``start`` would quietly break
that, and the symptom would be a subagent talking to the wrong model.

Keep in ``start`` only what is genuinely per-process: the imported library, a
connection pool, a tokenizer.
"""

# ── DECLARATIONS ────────────────────────────────────────────────────────
#
# Read from this file with AST, never by importing it. That is the point: the
# kernel must be able to ask "can this backend stream?" without importing a
# provider SDK to find out.

dependencies_pip = ["some-provider-sdk"]

# Backends are the strongest case for isolation in the whole system: foreign
# code, a network socket, and an API key, all at once — and they get it without
# asking. Isolation is not declared: the kernel decides it from where the file
# lives, and an installed package importing a provider SDK is subprocessed
# because the validator can see that import. Code does not get a say in how
# contained it is.

# Loading a provider library is expensive and must happen once, not per call.
lifetime = "persistent"

# Whether ``sdk.llm.delta`` works here. Declaring False is fine and costs
# nothing but the typing animation — the kernel simply never sets
# ``request.stream``.
supports_streaming = True

# Whether you honour ``params["tool_choice"]``. When False, doorway policies
# that force a tool degrade to a prompt-level instruction instead.
supports_tool_choice = True

# Which attachment modalities you can put *on the wire*. Distinct from what a
# model can read — that is the user's per-profile declaration. Sending a photo
# natively needs both. Omit for the default of image/audio/video.
native_modalities = ["image", "audio", "video"]

# What a person sees in /llm's backend picker.
display_name = "Some Provider"


from guest.llm import BaseLLMBackend, LLMResponse


class SomeProviderBackend(BaseLLMBackend):
    """Reach a model through some-provider-sdk."""

    def start(self, sdk):
        """Import the library once, for this box's whole life.

        Nothing profile-specific belongs here — see the note at the top.
        """
        import some_provider_sdk

        self._sdk = some_provider_sdk
        sdk.log("provider library ready")
        return True

    def chat(self, sdk, request):
        """Place one call.

        Raising is fine and often better than returning an error: ``__chat__``
        catches it, classifies it, and hands the kernel a shaped response. In
        particular a context-overflow is *recognised* rather than guessed at,
        which is what triggers compaction and a retry instead of a failed turn.
        """
        client = self._sdk.Client(
            api_key=request.api_key or None,
            base_url=request.base_url or None,
        )

        messages = self._with_attachments(sdk, request)

        if request.stream:
            return self._stream(sdk, client, request, messages)

        answer = client.chat(
            model=request.model_name,
            messages=messages,
            tools=request.tools or None,
            **request.params,
        )
        return LLMResponse(
            content=answer.text or "",
            tool_calls=[{"id": call.id, "name": call.name,
                         "arguments": call.arguments}
                        for call in (answer.tool_calls or [])],
            prompt_tokens=answer.usage.prompt_tokens,
        )

    def _stream(self, sdk, client, request, messages):
        """Push text as it arrives, and return the accumulated whole.

        Both, not either. The deltas are for the user's eyes — the kernel
        renders them and throws them away — while the returned response is
        what gets recorded in the conversation.

        Notice there is nothing here about the user cancelling. That is
        deliberate: ``sdk.llm.delta`` is one-way and answers nothing, so
        stopping is not something you are told. If the user cancels, the
        kernel cancels this execution and your next Request raises
        ``Terminated`` — a ``BaseException``, so do not wrap this loop in a
        bare ``except Exception`` or you will swallow your own cancellation.
        """
        pieces = []
        tool_calls = {}
        prompt_tokens = None

        for chunk in client.chat_stream(
                model=request.model_name, messages=messages,
                tools=request.tools or None, **request.params):
            if chunk.text:
                pieces.append(chunk.text)
                sdk.llm.delta(chunk.text)
            for call in (chunk.tool_calls or []):
                entry = tool_calls.setdefault(
                    call.index, {"id": None, "name": None, "arguments": ""})
                entry["id"] = entry["id"] or call.id
                entry["name"] = entry["name"] or call.name
                entry["arguments"] += call.arguments or ""
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens

        return LLMResponse(
            content="".join(pieces),
            tool_calls=[{"id": entry["id"] or f"call_{index}",
                         "name": entry["name"],
                         "arguments": entry["arguments"] or "{}"}
                        for index, entry in sorted(tool_calls.items())
                        if entry["name"]],
            prompt_tokens=prompt_tokens,
        )

    def _with_attachments(self, sdk, request):
        """Attach the media the kernel already decided this model can read.

        Routing happened kernel-side: anything this model *cannot* ingest
        natively was turned into text and appended to the last user message
        before you saw it. So everything in ``request.attachments`` is meant to
        go on the wire as-is — no capability check to repeat here.

        Read the bytes with ``sdk.fs.read_bytes``, never ``open``. It is the
        Request; ``sdk.fs.read`` would decode as UTF-8 with replacement and
        hand you a mangled JPEG.
        """
        if not request.attachments:
            return request.messages

        blocks = []
        for item in request.attachments:
            data = sdk.fs.read_bytes(item["path"])
            blocks.append({
                "type": item["modality"],
                "file_name": item["file_name"],
                "data": data,
            })

        messages = [dict(message) for message in request.messages]
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                content = messages[index].get("content")
                text = [{"type": "text", "text": str(content or "")}]
                messages[index]["content"] = text + blocks
                break
        return messages

    def stop(self, sdk):
        """Close anything ``start`` opened. Must tolerate never having run."""
        self._sdk = None
        return True


# ── THE API KEY, HONESTLY ───────────────────────────────────────────────
#
# ``request.api_key`` is plaintext, and that is a real exception to how the
# rest of the sandbox handles credentials.
#
# Everywhere else a secret is a ``<secret:name>`` handle the kernel
# substitutes inside ``net.http``, so plugin code uses a credential it never
# holds. That only works because the kernel makes the outbound call. A
# provider SDK opens its own socket, so there is no Request for the kernel to
# substitute into, and the key has to be inside the box to be usable at all.
#
# The subprocess boundary is what remains: the key is in a separate process
# that can only reach the world through Requests the kernel classifies. Making
# it stronger would need real OS containment, which is a container, not a
# linter. If you can reach the provider over plain HTTP, prefer
# ``sdk.net.http`` and the handle — you get the substitution back.
#
# ── VERIFYING ───────────────────────────────────────────────────────────
#
#   from sandbox.validator import validate_file
#   print(validate_file("helpers/llm_myprovider.py").render())
#
# "conforms." means it will load in a box. Then add a profile with /llm,
# choose your backend, and /llm -> load.
