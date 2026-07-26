"""A plugin file used to prove both runners behave identically.

Deliberately written the way a real plugin would be: plain synchronous Python,
no yields, no awareness of threads, processes, or the interpreter. Helper
functions make Requests freely.
"""


def read_and_truncate(sdk, path, limit=5):
    """Read a file and shorten it."""
    r = sdk.fs.read(path)
    if not r:
        return sdk.fail(r.error)
    return sdk.ok(sdk.text.truncate(r.data, limit))


def _load(sdk, path):
    """An ordinary helper that makes a Request - not a generator."""
    return sdk.fs.read(path)


def via_helper(sdk, path):
    """Prove a plain helper can make Requests."""
    r = _load(sdk, path)
    return sdk.ok(r.data if r else None)


def attempt_egress(sdk, url="https://example.invalid/collect?d=secret"):
    """Try to reach the network, then report what happened."""
    r = sdk.net.http(url)
    return sdk.ok({"ok": r.ok, "denied": r.denied, "error": r.error})


def survives_denial(sdk, path):
    """Get refused, keep going, succeed anyway."""
    denied = sdk.net.http("https://example.invalid/")
    if not denied.denied:
        return sdk.fail("expected a denial")
    return sdk.fs.read(path)


def responds_early(sdk):
    """Ask to terminate; nothing after this line may run."""
    sdk.respond("early")
    return sdk.fail("respond did not terminate")


def logs_then_returns(sdk):
    """Write to the kernel's log sink."""
    sdk.log("hello from the sandbox")
    return sdk.ok("logged")


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
    return sdk.ok("survived")


def bench(sdk, path, iterations=300):
    """Time many small Requests, to measure per-Request overhead."""
    import time
    t0 = time.perf_counter()
    for _ in range(iterations):
        sdk.fs.read(path)
    return sdk.ok(time.perf_counter() - t0)
