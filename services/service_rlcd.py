"""On-demand typed decisions. No subscriptions, actions, or online learning.

The service contract is evaluate(state, questions, model=None). The initial
wire adapter is TypeSafe System One; compatible local servers use the same
contract. Future incompatible providers belong in _request, not callers.
Credentials stay SDK handles. Provider errors never echo submitted state.

validate(questions, state=None) answers the same checks evaluate makes before
I/O, as a verdict rather than an exception, so a caller holding questions an
LLM wrote can hand the reason straight back to it.
"""
import json
import math
from urllib.parse import urlsplit
from guest.bases import BaseService

# Jev's documented request budget: 64k tokens in total, of which the state
# plus the longest single question may use 32k. Characters/4 is an estimate,
# deliberately generous to the caller — the provider's refusal is the
# authority; this only catches the questions that could never fit.
MAX_TOTAL_TOKENS = 64_000
MAX_STATE_PLUS_QUESTION_TOKENS = 32_000


def _tokens(value) -> int:
    """Rough token count of a JSON-able value."""
    text = value if isinstance(value, str) else json.dumps(value)
    return len(text) // 4 + 1


class RLCD(BaseService):
    name = "rlcd"
    description = "Typed probabilistic decisions via Jev or a compatible endpoint."
    exports = ["status", "evaluate", "validate"]
    requests = ["net.http", "config.read", "config.write"]
    config_settings = [
        ("Endpoint", "rlcd_endpoint", "Full System One evaluation URL.", "https://api.typesafe.ai/v1/systemone", {"type": "text"}),
        ("Model", "rlcd_model", "Default model ID; pin a version for repeatable experiments.", "jev-1.13.0", {"type": "text"}),
        ("API key", "secret_rlcd_typesafe_api_key", "Optional on unauthenticated local endpoints.", "", {"type": "text"}),
    ]

    def on_install(self, sdk):
        """Allow the endpoint's host, so callers without a person watching can reach it.

        Jev is called from hooks and tasks, whose chains are unattended, and an
        unattended request to a host outside ``net_allowed_hosts`` is refused
        rather than asked. Installing is the one moment the question can be put
        to somebody. Read-then-skip, so an update never re-asks or edits a list
        the user has since changed. Pointing ``rlcd_endpoint`` at another host
        later means allowing that host yourself.
        """
        endpoint = sdk.config.read("rlcd_endpoint") or "https://api.typesafe.ai/v1/systemone"
        host = (urlsplit(endpoint).hostname or "").lower()
        if not host:
            return
        current = list(sdk.config.read("net_allowed_hosts") or [])
        if any(host == h or host.endswith("." + h)
               for h in (str(item).strip().lower() for item in current) if h):
            return
        try:
            sdk.config.write("net_allowed_hosts", [*current, host])
        except sdk.Failed as error:
            raise RuntimeError(f"{host} not allowed ({error}) — Jev calls from hooks "
                               "and tasks will be refused") from None
        sdk.log(f"{host} added to net_allowed_hosts")

    def start(self, sdk):
        return True

    def stop(self, sdk):
        pass

    def _settings(self, sdk):
        endpoint = sdk.config.read("rlcd_endpoint") or "https://api.typesafe.ai/v1/systemone"
        key = sdk.config.read("secret_rlcd_typesafe_api_key")
        parsed = urlsplit(endpoint)
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query
                or parsed.fragment or (parsed.scheme != "https" and not
                    (parsed.scheme == "http" and local))):
            raise ValueError("Endpoint must use HTTPS (HTTP allowed on loopback), without credentials, query, or fragment.")
        if not local and not key:
            raise ValueError("RLCD API key is missing.")
        return endpoint, key, local

    def status(self, sdk):
        """Configuration check only; makes no network call and returns no secrets."""
        endpoint, key, local = self._settings(sdk)
        return {"endpoint": endpoint,
                "model": sdk.config.read("rlcd_model") or "jev-1.13.0",
                "credential_status": "configured" if key else "not required for local endpoint",
                "configuration_valid": True,
                "protocol": "typesafe-systemone", "local": local,
                "online_learning": False}

    def _structured(self, value):
        return isinstance(value, (str, dict, list))

    def _validate_questions(self, state, questions):
        if not self._structured(state):
            raise ValueError("state must be a string, object, or array.")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object.")
        for name, q in questions.items():
            if not isinstance(name, str) or not name or not isinstance(q, dict):
                raise ValueError("Each question needs a nonempty string ID and object definition.")
            if not self._structured(q.get("instructions")):
                raise ValueError("Each question needs string/object/array instructions.")
            kind, criteria = q.get("type"), q.get("criteria")
            if kind == "choice":
                if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                    raise ValueError("choice criteria must contain 1..255 options.")
                if any(not isinstance(k, str) or not k or (v is not None and not self._structured(v)) for k, v in criteria.items()):
                    raise ValueError("Invalid choice criteria.")
            elif kind == "score":
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10 or any(not self._structured(v) for v in criteria):
                    raise ValueError("score criteria must contain 2..10 level descriptions.")
            elif kind == "noul":
                if criteria is not None and (not isinstance(criteria, dict) or any(k not in ("true", "false") or not self._structured(v) for k, v in criteria.items())):
                    raise ValueError("noul criteria may describe true and false only.")
            else:
                raise ValueError("Question type must be choice, score, or noul.")
        # Reject NaN, unsupported objects, and other non-JSON inputs before I/O.
        json.dumps({"state": state, "questions": questions}, allow_nan=False)

    def _number(self, value, low, high):
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high

    def _validate_response(self, data, questions):
        if not isinstance(data, dict) or not isinstance(data.get("model"), str) or not data["model"]:
            raise ValueError("Missing response model.")
        answers = data.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("Response question IDs do not match.")
        for name, q in questions.items():
            a = answers[name]
            kind = q["type"]
            if not isinstance(a, dict) or a.get("type") != kind:
                raise ValueError("Response answer type does not match.")
            if kind == "noul":
                if not self._number(a.get("noul"), 0, 1):
                    raise ValueError("Invalid yes/no probability.")
                continue
            expected = set(q["criteria"]) if kind == "choice" else set(str(i) for i in range(len(q["criteria"])))
            probs = a.get("probabilities")
            if (not isinstance(probs, dict) or set(probs) != expected
                    or any(not self._number(p, 0, 1) for p in probs.values())
                    or abs(sum(probs.values()) - 1) > 0.001
                    or not self._number(a.get("confidence"), 0, 1)):
                raise ValueError("Invalid probability distribution or confidence.")
            if kind == "choice":
                chosen = a.get("choice")
                if not isinstance(chosen, str) or chosen not in expected or probs[chosen] < max(probs.values()) - 0.001:
                    raise ValueError("Invalid selected option.")
            else:
                if not self._number(a.get("score"), 0, len(expected) - 1):
                    raise ValueError("Invalid score.")
                if abs(a["score"] - sum(int(k) * p for k, p in probs.items())) > 0.01:
                    raise ValueError("Score does not match its distribution.")
                if not isinstance(a.get("legend"), dict) or set(a["legend"]) != expected:
                    raise ValueError("Invalid score legend.")
        usage = data.get("usage")
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
            raise ValueError("Invalid usage metadata.")

    def _request(self, sdk, endpoint, key, payload):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return sdk.net.http(endpoint, method="POST", headers=headers, json=payload)

    def validate(self, sdk, questions, state=None):
        """Return {ok: True} or {ok: False, reason} for questions and a state.

        Never raises and makes no network call. ``state`` defaults to a short
        placeholder; pass one the size of the real states to have the budget
        checked against them.
        """
        state = "x" if state is None else state
        try:
            self._validate_questions(state, questions)
        except (ValueError, TypeError) as error:
            return {"ok": False, "reason": str(error)}
        state_tokens = _tokens(state)
        sizes = {name: _tokens(q) for name, q in questions.items()}
        longest = max(sizes, key=sizes.get)
        if state_tokens + sizes[longest] > MAX_STATE_PLUS_QUESTION_TOKENS:
            return {"ok": False, "reason":
                    f"State (~{state_tokens} tokens) plus question '{longest}' "
                    f"(~{sizes[longest]}) exceeds Jev's {MAX_STATE_PLUS_QUESTION_TOKENS}-token "
                    "limit for state plus one question. Shorten that question or the state."}
        total = state_tokens + sum(sizes.values())
        if total > MAX_TOTAL_TOKENS:
            return {"ok": False, "reason":
                    f"State plus all questions is ~{total} tokens, over Jev's "
                    f"{MAX_TOTAL_TOKENS}-token request limit. Split the questions "
                    "across two requests."}
        return {"ok": True}

    def evaluate(self, sdk, state, questions, model=None):
        """Return {ok, provider, model, answers, usage} or {ok:false, error}.

        Invalid inputs/config raise ValueError before I/O. HTTP/transport/schema
        failures return no answers. No automatic retries: callers schedule backoff
        using retryable/retry_after, avoiding sleeping inside a serialized service.
        Confidence is a distribution statistic, not permission to execute actions.
        """
        self._validate_questions(state, questions)
        endpoint, key, local = self._settings(sdk)
        selected = model if model is not None else (sdk.config.read("rlcd_model") or "jev-1.13.0")
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("model must be a nonempty string.")
        try:
            response = self._request(sdk, endpoint, key, {"model": selected, "state": state, "questions": questions})
        except Exception:
            return {"ok": False, "error": {"kind": "transport", "message": "Request failed or was denied; check network permissions and connectivity.", "retryable": False}}
        status = response.get("status")
        if status != 200:
            headers = response.get("headers") or {}
            retry_after = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
            return {"ok": False, "error": {"kind": "http", "status": status,
                    "message": "Provider returned a non-success HTTP status.",
                    "retryable": status in (429, 500, 502, 503, 504, 529), "retry_after": retry_after}}
        try:
            if response.get("truncated"):
                raise ValueError("Truncated response.")
            data = json.loads(response["body"])
            self._validate_response(data, questions)
        except (ValueError, TypeError, KeyError, OverflowError):
            return {"ok": False, "error": {"kind": "invalid_response", "message": "Provider response failed schema validation.", "retryable": False}}
        return {"ok": True, "model": data["model"],
                "answers": data["answers"], "usage": data["usage"]}
