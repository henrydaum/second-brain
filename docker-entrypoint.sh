#!/bin/sh
#
# What a container has to settle before the app starts, and a small vocabulary
# of ways to start it. Everything here is *image* policy — none of it is a
# kernel knob, and the app itself is unaware any of it happened.
#
#   run [args]     the app (default)
#   test [args]    the test suite, arguments passed to pytest
#   shell [args]   an interactive shell in the app tree
#   <anything>     executed as given: python -c ..., git log, pip list
#
set -eu

DATA_ROOT="${XDG_DATA_HOME:-/data}"
DATA_DIR="${DATA_ROOT}/Second Brain"
SEED_DIR="${SB_SEED_DIR:-/seed}"

# --- 1. The data tree has to be writable, and has to say so clearly ---------
#
# Everything that survives the container lives here. Left to fail on its own
# this arrives as a traceback out of DATA_DIR.mkdir in main.pyw, several
# screens from the thing that actually went wrong.
if ! mkdir -p "${DATA_DIR}" 2>/dev/null || [ ! -w "${DATA_DIR}" ]; then
    echo "second-brain: ${DATA_ROOT} is not writable by uid $(id -u)." >&2
    echo >&2
    echo "  A named volume is the easy path — Docker gives it the image's" >&2
    echo "  ownership on first use:" >&2
    echo "      docker run -v sb-data:/data second-brain" >&2
    echo >&2
    echo "  A host directory keeps its host ownership, so build the image for" >&2
    echo "  that user, or hand the directory to uid $(id -u):" >&2
    echo "      docker build --build-arg UID=\$(id -u) --build-arg GID=\$(id -g) -t second-brain ." >&2
    exit 1
fi

# --- 2. A seeded data tree ------------------------------------------------
#
# How a run arrives with its config already made: bake or mount a prepared
# tree at /seed and it is copied in the first time DATA_DIR is empty. This is
# what makes a benchmark image a two-line derivation of this one (a golden
# template plus --rm, so every run starts from the same known state), and it
# is equally how you ship a preconfigured container to someone else.
#
# config.json is the sentinel because it is what the kernel itself writes on
# first boot: present means this tree has been lived in, and seeding over it
# would overwrite a real installation.
if [ -d "${SEED_DIR}" ] && [ ! -e "${DATA_DIR}/config.json" ]; then
    echo "second-brain: seeding ${DATA_DIR} from ${SEED_DIR}" >&2
    cp -a "${SEED_DIR}/." "${DATA_DIR}/"
fi

# --- 3. The HTTP frontend, if this container publishes one ------------------
#
# The kernel binds 127.0.0.1 on purpose and calls exposure the tunnel
# operator's decision. Inside a container, loopback is not reachable from
# `docker run -p` at all — the port is published and nothing answers, which
# looks exactly like a broken app. So the tunnel is one forwarder, opt-in:
#
#   -e SB_EXPOSE_HTTP=8787 -p 8787:8787
#
# The listener stays behind the frontend's bearer token; publishing it puts
# that token between your LAN and your assistant, so make it a long one.
#
# It binds the container's *own* address rather than 0.0.0.0, and that is the
# difference between this working and not. A wildcard listener on 8787 owns
# loopback 8787 too, so the app's own bind then fails with EADDRINUSE — and it
# fails inside the frontend, where it reads as "the HTTP frontend is broken"
# with nothing pointing back at the forwarder. Naming one interface leaves
# loopback free, and -p reaches it because that is the address Docker's rule
# forwards to.
if [ -n "${SB_EXPOSE_HTTP:-}" ]; then
    app_port="${SB_HTTP_PORT:-8787}"
    bind_ip="$(hostname -i 2>/dev/null | tr ' ' '\n' | grep -v '^127\.' | head -n 1)"
    if [ -z "${bind_ip}" ]; then
        echo "second-brain: SB_EXPOSE_HTTP is set but this container has no" >&2
        echo "              non-loopback address to forward from. Did you run" >&2
        echo "              it with --network none or --network host?" >&2
        exit 1
    fi
    echo "second-brain: forwarding ${bind_ip}:${SB_EXPOSE_HTTP} to 127.0.0.1:${app_port}" >&2
    socat "TCP-LISTEN:${SB_EXPOSE_HTTP},bind=${bind_ip},fork,reuseaddr" \
          "TCP:127.0.0.1:${app_port}" &
fi

# --- 4. Say what a headless container is about to look like -----------------
#
# ``enabled_frontends`` ships as ["repl"], and the REPL claims the container's
# stdin. With no terminal attached it reads EOF, closes the console and stops
# itself — correct, and it narrates the process over five WARNING lines and two
# ERRORs that read exactly like a crash to anyone reading ``docker logs`` for
# the first time. The app is fine; it simply has no frontend. Saying so first
# costs a line and turns an alarming log into an expected one.
# The test is /dev/null specifically, not "is stdin a terminal". ``docker run``
# without -i hands the container /dev/null, which is EOF on the first read —
# that is the case this warns about. With -i and no -t, stdin is a pipe that
# stays open and the REPL runs perfectly well on it, which is how a script
# drives one; warning there would promise a failure that never arrives.
if [ "${1:-run}" = "run" ] && [ ! -t 0 ] \
   && [ "$(readlink /proc/self/fd/0 2>/dev/null)" = "/dev/null" ]; then
    echo "second-brain: no terminal on stdin — the REPL frontend will stop itself" >&2
    echo "              shortly, with warnings that look worse than they are. Use" >&2
    echo "              'docker run -it' for a REPL; otherwise configure a" >&2
    echo "              frontend that needs no console (Telegram, HTTP)." >&2
fi

# --- 5. What to run --------------------------------------------------------
#
# exec in every branch: the app installs SIGTERM and SIGINT handlers and
# shuts down cleanly, which it only ever gets to do if it is PID 1 rather than
# a child of this script.
case "${1:-run}" in
    run)
        if [ "$#" -gt 0 ]; then shift; fi
        exec python main.py "$@"
        ;;
    test)
        shift
        exec python -m pytest "$@"
        ;;
    shell)
        shift
        exec /bin/bash "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
