"""What a profile's provider parameters actually become on the wire.

The kernel puts ``reasoning_effort`` on every call whose profile did not say
otherwise (``llm.registry.DEFAULT_REASONING_EFFORT``). It never reaches a
provider in that spelling. A backend *translates* it, and LiteLLM decides how
from the **model name** — so one string, chosen to name a model, silently
decides three different outcomes:

* a name whose first path segment matches a provider LiteLLM knows gets that
  provider's dialect (``deepseek/…`` sends ``thinking={"type": "enabled"}``,
  ``ollama/…`` sends ``think=True``) — addressed to whatever host the
  profile's endpoint actually points at, which is very often not that
  provider. An endpoint that has never heard of the field answers 400.
* a name it does not recognise is served over the OpenAI-compatible path,
  where ``drop_params`` **discards** the parameter. The call succeeds and the
  setting does nothing. ``/llm`` goes on showing the level you picked.
* anything the endpoint does understand goes through unchanged.

None of the three is visible from ``/llm``, and the failure of the first is
whatever the endpoint says — measured against one aggregator, the entire
refusal was ``{'code': 400, 'msg': 'bad request'}``.

So this reports, per profile, which of the three you are in. It is offline by
default: no key is used, nothing is sent, and the answer comes from the same
LiteLLM translation the backend will perform. ``--live`` adds one real
five-token call per profile, which is the only way to see the endpoint's own
verdict on what we would send it.

    python dev/probe_params.py                       # every configured profile
    python dev/probe_params.py --profile deepseek-ai/DeepSeek-V4-Pro
    python dev/probe_params.py --live                # +1 tiny call per profile

Run it on the deployment that shows the symptom — the endpoint and the model
name are both part of the question, so the answer is allowed to differ per
machine.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RULE = "=" * 74
THIN = "-" * 74

#: The set ``llm_litellm._KNOWN_PROVIDER_PREFIXES`` holds, used only if that
#: file cannot be read. A fallback rather than the source of truth: the backend
#: is an installed package and may have moved on, and a probe reporting a stale
#: rule as fact is worse than one saying it could not find the file.
_FALLBACK_PREFIXES = {
    "anthropic", "azure", "bedrock", "cohere", "deepseek", "gemini", "groq",
    "minimax", "mistral", "ollama", "openai", "openrouter", "vertex_ai", "xai",
}


def known_prefixes():
    """``_KNOWN_PROVIDER_PREFIXES`` as the installed backend declares it.

    Read with AST rather than imported, for the reason the registry reads
    every other backend declaration that way: importing this file means
    importing litellm into a process that has no box to hold it.
    """
    import trees

    for _root, backends in trees.dirs_for("llm"):
        source_file = backends / "llm_litellm.py"
        if not source_file.exists():
            continue
        try:
            tree = ast.parse(source_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_KNOWN_PROVIDER_PREFIXES" not in names:
                continue
            try:
                return set(ast.literal_eval(node.value)), str(source_file)
            except ValueError:
                break
    return _FALLBACK_PREFIXES, "(built-in fallback - backend file not found)"


def model_for_litellm(name, base_url, prefixes):
    """The same string ``llm_litellm._model_name`` builds.

    Borrowed rather than imported, like ``probe_reasoning._key``: the rule is
    four lines and the file it lives in cannot be imported here. What it is
    borrowed *from* is checked above — the prefix set is read from the real
    backend, and the prefix set is the whole of what varies.
    """
    provider = name.split("/", 1)[0].lower().replace("-", "_") if "/" in name else ""
    if base_url and provider not in prefixes:
        return "openai/" + name
    return name


def translate(model, params):
    """What LiteLLM turns *params* into for *model*, and the provider it chose."""
    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    resolved, provider, _key_, _base = litellm.get_llm_provider(model=model)
    sendable = {k: v for k, v in params.items() if k != "tool_choice"}
    out = litellm.utils.get_optional_params(
        model=resolved, custom_llm_provider=provider, **sendable)
    # ``stream`` is added by the translation itself and says nothing about
    # what the profile asked for.
    out.pop("stream", None)
    if out.get("extra_body") == {}:
        out.pop("extra_body")
    return out, provider


def verdicts(sent, received):
    """``(param, verdict, detail)`` for each parameter the profile sends."""
    flat = dict(received)
    body = flat.pop("extra_body", None)
    if isinstance(body, dict):
        flat.update(body)
    rows = []
    for key, value in sorted(sent.items()):
        if key in flat and flat[key] == value:
            rows.append((key, "KEPT", "sent as %s=%r" % (key, value)))
        elif key in flat:
            rows.append((key, "CHANGED", "sent as %s=%r" % (key, flat[key])))
        else:
            # Whatever the translation produced that the profile did not name
            # is where this parameter went, if it went anywhere.
            extra = {k: v for k, v in flat.items() if k not in sent}
            if extra:
                rows.append((key, "TRANSLATED", "sent as " + ", ".join(
                    "%s=%r" % (k, v) for k, v in sorted(extra.items()))))
            else:
                rows.append((key, "DROPPED", "never reaches the endpoint"))
    return rows


def _key(profile):
    """The resolution ``Brain.api_key`` performs, borrowed as in the other probe."""
    import os

    raw = (profile.get("secret_llm_api_key")
           or profile.get("llm_api_key", "") or "")
    return os.environ.get(raw, raw) if raw else ""


def live_call(model, profile, params):
    """One five-token call, reported as the endpoint's own verdict."""
    import litellm

    kwargs = dict(params)
    kwargs.pop("tool_choice", None)
    if profile.get("llm_endpoint"):
        kwargs["api_base"] = profile["llm_endpoint"]
    kwargs["api_key"] = _key(profile)
    try:
        litellm.completion(model=model, max_tokens=5,
                           messages=[{"role": "user", "content": "Say OK."}],
                           **kwargs)
    except Exception as exc:                    # noqa: BLE001 — a probe reports
        return "REFUSED  %s: %s" % (type(exc).__name__, str(exc)[:400])
    return "ACCEPTED  the endpoint took these parameters"


def report(name, brain, prefixes, live):
    """One profile, as facts first and a verdict last."""
    params = brain.params
    defaults = brain.default_params
    base_url = brain.base_url
    model = model_for_litellm(name, base_url, prefixes)

    print("\n%s\n%s\n%s" % (THIN, name, THIN))
    print("  endpoint        %s" % (base_url or "(provider default)"))
    print("  backend         %s" % brain.backend_name)

    if not params:
        print("  profile sends   (nothing - every parameter is null)")
        return

    print("  profile sends   " + ", ".join(
        "%s=%r%s" % (k, v, "  <- kernel default, not set by this profile"
                     if k in defaults else "")
        for k, v in sorted(params.items())))

    try:
        received, provider = translate(model, params)
    except Exception as exc:                    # noqa: BLE001 — a probe reports
        print("  could not translate: %s: %s" % (type(exc).__name__, exc))
        return

    print("  litellm sees    %s   (provider: %s)" % (model, provider))
    print("  endpoint gets   %s" % (received or "(nothing)"))
    print()
    rows = verdicts(params, received)
    for key, verdict, detail in rows:
        flag = "  <- kernel default" if key in defaults else ""
        print("    %-20s %-11s %s%s" % (key, verdict, detail, flag))

    translated = [k for k, v, _ in rows if v == "TRANSLATED"]
    dropped = [k for k, v, _ in rows if v == "DROPPED"]
    print("\n  => ", end="")
    if translated:
        print("TRANSLATED. %s is being rewritten into %s's own\n"
              "         dialect, and sent to\n             %s\n"
              "         If that host is not %s it may well refuse the call - "
              "and its\n         refusal will not mention %s. Set it to null "
              "in Extra\n         parameters, or rename the profile so its "
              "prefix is not a\n         LiteLLM provider name."
              % (", ".join(translated), provider,
                 base_url or "(the provider default)", provider,
                 translated[0]))
    elif dropped:
        print("DROPPED. %s never leaves this machine - drop_params\n"
              "         discards it, because LiteLLM's %s config does not "
              "list it as\n         supported. Nothing breaks and nothing "
              "happens: the level /llm\n         shows is inert. If this "
              "endpoint takes reasoning some other\n         way, spell that "
              "way directly in Extra parameters."
              % (", ".join(dropped), provider))
    else:
        print("FORWARDED. Everything this profile sets reaches the endpoint "
              "as\n         written.")

    if live:
        print("\n  live: %s" % live_call(model, brain.profile, params))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="probe only this profile")
    parser.add_argument("--live", action="store_true",
                        help="also make one five-token call per profile")
    args = parser.parse_args(argv)

    import llm
    from config import config_manager

    config = config_manager.load()
    configured = config.get("llm_profiles") or {}
    if args.profile:
        if args.profile not in configured:
            raise SystemExit("no such profile: %s\nconfigured: %s"
                             % (args.profile, ", ".join(sorted(configured))))
        configured = {args.profile: configured[args.profile]}
    if not configured:
        raise SystemExit("no LLM profiles configured - run /setup first.")

    prefixes, source = known_prefixes()

    print(RULE)
    print("PROVIDER PARAMETER PROBE")
    print("offline%s; prefix rule from %s"
          % (" + one live call per profile" if args.live else "", source))
    print(RULE)

    for name, profile in sorted(configured.items()):
        report(name, llm.registry.Brain(name, profile, config), prefixes,
               args.live)

    print("\n%s" % RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
