"""Starting the web app, and saying where it ended up.

``frame_ui/`` is part of this repo, so the kernel can treat it the way it
treats a frontend: something it owns and therefore something it starts. What
makes that worth doing rather than leaving to a second terminal is the failure
it removes — a UI that is one forgotten ``npm run dev`` away from working looks
exactly like a UI that is broken.

Three steps, in an order that matters:

1. **Probe first.** Something may already be serving — a dev server left
   running from before a ``/restart``, or a real deployment (the macOS install
   serves a build through Caddy and has no dev server at all). Starting a
   second one would bind-fail at best and fight over the port at worst, so a
   reachable address is taken as the answer and nothing is spawned.
2. **Start it, if nothing answered and the user wants that.**
3. **Say where it is**, once it actually answers. The notification is the
   *point* — "it is running" is not useful, "open this" is — which is why it is
   raised on the first successful probe rather than when the process starts.
   A process that starts and then exits on a port conflict is exactly the case
   a start-time notice would get wrong.

Nothing here is fatal. A machine with no Node, a checkout with no
``node_modules``, a port already taken by something else: each is logged and
the kernel carries on, because the web UI is one way in and the REPL is
another.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from paths import DATA_DIR, ROOT_DIR
from runtime import notifications

logger = logging.getLogger("WebUI")

#: The app, in this repo.
UI_DIR = ROOT_DIR / "frame_ui"

#: Where the dev server's own output goes. Not the app log: npm and Vite are
#: noisy, chatty in colour codes, and the thing anybody wants from them is the
#: stack trace on the one day it will not start.
LOG_FILE = DATA_DIR / "web_ui.log"

#: How long to keep asking before giving up on the notification. A cold Vite
#: start is a second or two; a first run that has to build its dependency
#: cache is longer. Ninety seconds is generous enough to cover a slow laptop
#: and short enough that a server which is never coming stops being waited for.
READY_TIMEOUT = 90.0
PROBE_INTERVAL = 1.0

#: One probe's own patience. Short: this runs in a loop, and a socket that is
#: not listening answers immediately anyway — the timeout only bites when
#: something is listening but wedged, which is a case to retry rather than
#: block on.
PROBE_TIMEOUT = 2.0

_process: subprocess.Popen | None = None
_lock = threading.Lock()


# ── The probe ─────────────────────────────────────────────────────────

def reachable(url: str) -> bool:
    """Whether something is serving at ``url``.

    **Any HTTP answer counts, including 404 and 500.** The question is whether
    a server is there, not whether it likes the request — and during startup
    Vite answers before its own routes are ready. Treating a status code as a
    failure would mean waiting for a page this function is not entitled to
    have an opinion about.
    """
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


# ── Starting it ───────────────────────────────────────────────────────

def _port_of(url: str) -> str:
    """The port in ``url``, as a string, or ""."""
    try:
        return str(urlparse(url).port or "")
    except ValueError:
        return ""


def _spawn(url: str) -> subprocess.Popen | None:
    """Launch ``npm run dev``, or explain why not.

    The port is passed **in**, from ``ui_url``, rather than left to the dev
    server's own default. One address is configured and both halves read it;
    the alternative is a port in ``config.json`` and a port in ``.env.local``
    that agree right up until somebody changes one, and the symptom of that is
    a notification pointing at nothing.
    """
    if not UI_DIR.is_dir() or not (UI_DIR / "package.json").is_file():
        logger.info("No web app at %s; not starting one.", UI_DIR)
        return None
    if not (UI_DIR / "node_modules").is_dir():
        logger.warning(
            "The web UI has no node_modules. Run `npm install` in %s, then "
            "restart. Not doing it here: a first install is minutes long and "
            "nobody asked for one at boot.", UI_DIR)
        return None

    env = dict(os.environ)
    if port := _port_of(url):
        env["VITE_UI_PORT"] = port

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
    except OSError:
        log = subprocess.DEVNULL

    try:
        # ``shell=True`` with a string: ``npm`` on Windows is ``npm.cmd``, a
        # batch file rather than an executable, so a bare exec fails outright
        # and reads as "npm is not installed" on a machine that has it.
        #
        # The process-group flags are what make this stoppable. A shell spawns
        # *node* as a child, so killing what we hold kills the shell and leaves
        # the dev server running on the port — after which the next boot probes
        # it, finds it reachable, and adopts an orphan from a previous life.
        return subprocess.Popen(
            "npm run dev",
            shell=True,
            cwd=str(UI_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **(
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt" else {"start_new_session": True}
            ),
        )
    except Exception:
        logger.exception("Could not start the web UI.")
        return None


def _announce(url: str) -> None:
    """The one line this whole module exists to produce."""
    notifications.notify(
        title=f"UI is reachable at: {url}",
        source="web_ui",
        level="success",
    )
    logger.info("UI is reachable at: %s", url)


def _watch(url: str, autostart: bool) -> None:
    """Probe, start if needed, probe again, announce. Runs on its own thread."""
    global _process

    if reachable(url):
        # Already served — a survivor of a /restart, or a real deployment.
        _announce(url)
        return

    if not autostart:
        logger.info("Nothing is serving %s and ui_autostart is off.", url)
        return

    with _lock:
        _process = _spawn(url)
    if _process is None:
        return
    logger.info("Starting the web UI (npm run dev in %s); output goes to %s",
                UI_DIR, LOG_FILE)

    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        # The exit check comes first so a server that died on a port conflict
        # is reported as that, rather than as ninety seconds of silence.
        if _process.poll() is not None:
            logger.warning("The web UI exited (code %s) before serving %s. "
                           "See %s.", _process.returncode, url, LOG_FILE)
            return
        if reachable(url):
            _announce(url)
            return
        time.sleep(PROBE_INTERVAL)

    logger.warning("The web UI did not answer at %s within %.0fs. See %s.",
                   url, READY_TIMEOUT, LOG_FILE)


def serve(config: dict) -> None:
    """Bring the web UI up, in the background. Safe to call when it cannot.

    Returns immediately: every part of this either waits on a socket or on
    npm, and boot must not.
    """
    url = str(config.get("ui_url") or "").strip()
    if not url:
        return
    threading.Thread(
        target=_watch,
        args=(url, bool(config.get("ui_autostart", True))),
        daemon=True,
        name="web-ui",
    ).start()


# ── Stopping it ───────────────────────────────────────────────────────

def stop() -> None:
    """End a dev server this process started. Never touches one it adopted.

    The group, not the process: what we hold is a shell, and the thing on the
    port is its child. ``taskkill /T`` and ``killpg`` are the two spellings of
    "and everything under it".
    """
    global _process
    with _lock:
        process, _process = _process, None
    if process is None or process.poll() is not None:
        return
    logger.info("Stopping the web UI...")
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=False)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception as exc:
        logger.debug("Web UI shutdown: %s", exc)
    try:
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
