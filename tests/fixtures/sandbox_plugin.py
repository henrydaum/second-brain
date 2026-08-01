"""A plugin file used to prove both runners behave identically.

Written the way a real plugin should be: Requests return their value, failures
raise, and the runner wraps a bare return. No result objects, no branches that
exist only to forward an error.
"""


def read_and_truncate(sdk, path, limit=5):
    """Read a file and shorten it."""
    return sdk.text.truncate(sdk.fs.read(path), limit)


def _load(sdk, path):
    """An ordinary helper that makes a Request - not a generator."""
    return sdk.fs.read(path)


def via_helper(sdk, path):
    """Prove a plain helper can make Requests."""
    return _load(sdk, path)


def attempt_egress(sdk, url="https://example.invalid/collect?d=secret"):
    """Try to reach the network, then report what happened."""
    try:
        sdk.net.http(url)
    except sdk.Denied as refused:
        return {"ok": False, "denied": True, "error": refused.error}
    return {"ok": True, "denied": False, "error": ""}


def survives_denial(sdk, path):
    """Get refused, keep going, succeed anyway."""
    try:
        sdk.net.http("https://example.invalid/")
        return sdk.fail("expected a denial")
    except sdk.Denied:
        return sdk.fs.read(path)


def responds_early(sdk):
    """Ask to terminate; nothing after this line may run."""
    sdk.respond("early")
    return sdk.fail("respond did not terminate")


def logs_then_returns(sdk):
    """Write to the kernel's log sink."""
    sdk.log("hello from the sandbox")
    return "logged"


def raises(sdk):
    """Break in a way the plugin did not anticipate."""
    raise ValueError("something went wrong")


def spins(sdk):
    """Never stop making Requests."""
    while True:
        sdk.fs.list(".")


def prints_to_stdout(sdk):
    """Print, which must not corrupt the protocol stream."""
    print("this must not reach the wire")
    return "survived"


def returns_every_extra(sdk):
    """Fill every optional field on Result, so the wire format is exercised.

    ``to_dict``/``from_dict`` enumerate fields by hand, and the in-process
    runner never serializes — so only a subprocess run can catch a field that
    was added to the dataclass and forgotten on the wire.
    """
    return sdk.ok(
        {"n": 1},
        llm_summary="a summary",
        attachments=["/tmp/a.png"],
        also_contains=["nested"],
        discovered_paths=["/tmp/found"],
    )


def returns_canonical_values(sdk):
    """Values whose Python spelling differs from their wire spelling."""
    return {"pair": (1, 2), "blob": bytearray(b"ab")}


def returns_reserved_tag_dict(sdk):
    """User data must not be mistaken for the bytes codec's envelope."""
    return {"__bytes__": "AA=="}


def returns_live_object(sdk):
    """A live object is outside the return contract."""
    return object()


def sends_live_object(sdk):
    """A live object is outside the Request contract too."""
    return sdk.services.call("absent", "noop", value=object())


def returns_oversized_value(sdk):
    """One Result must fit in one protocol frame under either runner."""
    return "x" * (17 * 1024 * 1024)


def sends_oversized_request(sdk):
    """One Request must fit in one protocol frame under either runner."""
    return sdk.services.call("x" * (17 * 1024 * 1024), "noop")
