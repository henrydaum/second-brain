# syntax=docker/dockerfile:1
#
# Second Brain in a container.
#
# One image, several jobs: an interactive REPL, a headless service behind a
# transport, a benchmark harness's task container, and the test suite. What
# separates those is never the image — it is the *data tree*, which lives
# outside it. DATA_DIR holds the config, the database and the plugin trees, so
# everything specific to a run (which model, which frontend, which packages)
# arrives there and the image stays general.
#
#   docker build -t second-brain .
#   docker run --rm -it --init -v sb-data:/data second-brain
#
# Headless runs, publishing the HTTP port, seeding a prepared data tree and
# deriving a benchmark image are all in docs/DOCKER.md.

# 3.11 is the floor the app supports. The default is the newest version the
# suite has been run on; drop to an older one when a store package you want
# has no wheels for it yet (torch and friends lag a release or two):
#   docker build --build-arg PYTHON_VERSION=3.13 -t second-brain .
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

# git    The package store *is* a git ref: ``store_backend`` shells out to
#        ``git ls-tree`` and ``git fetch``. A container without git boots
#        perfectly and can never install a package, and the failure surfaces
#        as a bare FileNotFoundError two layers down.
# tzdata Scheduled jobs are cron expressions. With no zone database every
#        schedule is UTC, which is not what "9am" meant when it was typed.
#        Pass -e TZ=America/New_York to pick one.
# socat  The kernel binds its HTTP listener to loopback deliberately —
#        exposure "belongs to whoever runs the tunnel", and in a container the
#        entrypoint is that tunnel. Without a forwarder, -p publishes a port
#        nothing is listening on. Opt in with SB_EXPOSE_HTTP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git socat tzdata \
    && rm -rf /var/lib/apt/lists/*

# Not root. The sandbox contains what plugin code may *ask* the kernel for, not
# what a process on this machine may do — the container's user is the outer
# half of that boundary. Match a host uid when bind-mounting a directory from
# it: --build-arg UID=$(id -u) --build-arg GID=$(id -g).
ARG UID=1000
ARG GID=1000
RUN set -eu; \
    getent group "${GID}" >/dev/null || groupadd --gid "${GID}" app; \
    getent passwd "${UID}" >/dev/null || useradd --uid "${UID}" --gid "${GID}" --home-dir /home/app --create-home --shell /bin/bash app; \
    mkdir -p /home/app; \
    chown "${UID}:${GID}" /home/app

WORKDIR /app

# The *directory*, not just what lands in it. ``COPY --chown`` owns the files
# it copies and says nothing about the folder WORKDIR already made as root, so
# reading the tree works and creating anything new in it does not — which
# surfaces a long way from here, as pytest failing to make its temp root.
RUN chown "${UID}:${GID}" /app

# Kernel dependencies only, and deliberately into the system site-packages as
# root so they are part of the image. Everything heavier belongs to a store
# package and installs at runtime — see PYTHONUSERBASE below.
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-user -r requirements.txt

# The app tree, ``.git`` included: the store ref lives there, and dropping it
# gives you a container that boots and can never install a package.
COPY --chown=${UID}:${GID} . .

# git refuses to read a repository whose owner is not the current user, which
# is exactly what a bind-mounted source tree looks like from in here — and it
# would present as an empty store rather than as an error.
RUN git config --system --add safe.directory /app

# The exec bit does not survive a build context checked out on Windows, and an
# entrypoint without it fails the container at start with "permission denied"
# and nothing else to go on.
RUN chmod +x /app/docker-entrypoint.sh

# ``paths.py`` resolves DATA_DIR from XDG_DATA_HOME on Linux, so the data tree
# is "/data/Second Brain". There is no first-class DATA_DIR override; this env
# var is the supported knob.
ENV HOME=/home/app \
    XDG_DATA_HOME=/data \
    PYTHONUNBUFFERED=1 \
    GIT_TERMINAL_PROMPT=0

# Store packages declare ``dependencies_pip`` and the package manager runs
# ``pip install`` for them at runtime. Sent to the system site-packages those
# would be root-owned and — worse — would vanish with the container while the
# plugin files needing them survive on the volume: a half-installed package
# that fails to load with nothing anywhere saying why. PYTHONUSERBASE puts them
# on the same volume as the plugins, so the two halves live and die together.
# Python adds that path to sys.path itself, in the app and in every subprocess
# box, because neither is started with -s or -E.
ENV PYTHONUSERBASE=/data/python \
    PIP_USER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Created here, with the right owner, so a named volume mounted at /data
# inherits it. A volume Docker has to create itself comes up owned by root.
#
# The user-site directory itself is made now rather than by the first pip run:
# Python only puts it on ``sys.path`` when it exists at interpreter start, so a
# tree that appears mid-run is invisible to the process that installed into it.
RUN mkdir -p /data /data/python "$(python -m site --user-site)" \
    && chown -R "${UID}:${GID}" /data

USER ${UID}:${GID}

# Bytecode for the app tree, baked in — and written as the app user, so a
# runtime recompile of a file that changed is allowed to replace it. Every
# sandboxed call in a subprocess box is a fresh ``python -m guest.child``, and
# a tree with no ``__pycache__`` recompiles the guest on each one. Failure here
# is a slower container, never a broken one, so it does not fail the build.
RUN python -m compileall -q /app >/dev/null 2>&1 || true

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Headless by default: which frontend runs is a config decision, not an image
# one. A REPL needs a terminal (docker run -it), a service container installs
# frontend_http and publishes it, a benchmark container names its own.
CMD ["run"]
