# Running Second Brain in Docker

One image, several jobs: a REPL you talk to, a headless service behind the web
UI or Telegram, a benchmark harness's task container, and the test suite on a
Linux that isn't your laptop. None of those is a different image — what
separates them is the **data tree**, which lives outside it.

If you have never used Docker, start at the top. If you have, skip to
[What this image decides](#what-this-image-decides).

---

## Docker in five minutes

Docker runs a program with its own filesystem, its own installed packages and
its own network, on a kernel it borrows from your machine. Not a virtual
machine — there is no second operating system booting — but from the program's
side it may as well be alone on a fresh Linux box.

Four nouns do almost all the work:

| | What it is | The analogy |
|---|---|---|
| **Image** | A frozen filesystem plus a default command. Read-only, built once, identical everywhere. | A class |
| **Container** | One running copy of an image, with a thin writable layer of its own. | An instance |
| **Volume** | Storage that outlives any container, mounted into one at a path. | A hard drive you plug in |
| **Port publish** | A hole from your machine's network into the container's. | Port forwarding on a router |

The two rules that catch everyone:

**A container's writable layer dies with the container.** `docker run` makes a
new one every time. Anything you want to keep — the database, your config, the
packages you installed — has to be on a volume, or it is gone the next time you
start. This is not a caveat, it is the point: it is what makes a container
reproducible.

**A container has its own network stack.** `localhost` inside the container is
the container, not your machine. That is why publishing a port is an explicit
act, and it is why the HTTP frontend needs one extra flag here — see
[Reaching the web UI](#reaching-the-web-ui).

The commands worth knowing:

```bash
docker build -t second-brain .        # read the Dockerfile, produce an image
docker run --rm -it second-brain      # start a container from it
docker ps                             # what is running
docker logs -f <name>                 # what it is saying
docker stop <name>                    # SIGTERM, then SIGKILL after 10s
docker exec -it <name> bash           # a shell inside something already running
```

The flags worth knowing: `--rm` deletes the container when it exits (otherwise
they pile up), `-it` connects your keyboard and gives it a terminal, `-d` runs
it in the background, `-v` mounts a volume, `-p` publishes a port, `-e` sets an
environment variable, and `--init` puts a real init process at PID 1 so
orphaned processes get reaped.

---

## The five-minute version

```bash
docker build -t second-brain .
docker run --rm -it --init -v sb-data:/data second-brain
```

You are in the REPL. Run `/setup`, connect a model, say hello. Quit with
`/quit`; the container is deleted and everything you did is still in the
`sb-data` volume, waiting for the next `docker run`.

Or, the same thing with the flags written down for you:

```bash
docker compose run --rm second-brain
```

---

## What this image decides

Six things, all of them container concerns rather than kernel ones. The app is
unaware any of them happened.

**The data tree is `/data`.** `paths.py` resolves `DATA_DIR` from
`XDG_DATA_HOME` on Linux, so the config, the database, the ledger and the three
plugin trees are all under `/data/Second Brain`. Mount something there or lose
your work on exit. A named volume (`-v sb-data:/data`) is the easy path; a host
directory (`-v $PWD/sb-data:/data`) lets you read the files with your own
editor, at the cost of the ownership dance below.

**It does not run as root.** The sandbox contains what plugin code may *ask*
the kernel for; the container user is the outer half of that. A named volume
inherits the image's ownership and Just Works. A **host directory** keeps its
host ownership, so build the image for yourself:

```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t second-brain .
```

**`.git` ships in the image, on purpose.** The package store is a git ref:
`/packages` reads `git ls-tree origin/store` inside the app tree. Strip `.git`
and you get a container that boots perfectly and can never install anything,
failing as a bare `FileNotFoundError` two layers down. `git` itself is
installed for the same reason.

**Runtime `pip` installs land on the volume.** Store packages declare
`dependencies_pip`, and the package manager `pip install`s them while the app
runs. Left in the image's site-packages they would vanish with the container
while the plugin files that need them survived on the volume — a package that
is installed according to `/packages list` and fails to load, with nothing
anywhere saying why. `PYTHONUSERBASE=/data/python` puts both halves on the same
volume.

**Time is UTC unless you say otherwise.** Scheduled jobs are cron expressions,
read in the container's zone. `-e TZ=America/New_York`.

**The HTTP listener is loopback-only**, which in a container means the
published port reaches nothing. Read on.

### The knobs

These belong to the image, not to the kernel — nothing in the app reads them.
Everything the *app* is configured by lives in `/config` and the data tree.

| Variable | Default | What it does |
|---|---|---|
| `XDG_DATA_HOME` | `/data` | Where the data tree goes: `$XDG_DATA_HOME/Second Brain`. |
| `TZ` | `UTC` | The zone cron schedules are read in. |
| `SB_SEED_DIR` | `/seed` | A prepared data tree, copied in on first boot only. |
| `SB_EXPOSE_HTTP` | unset | Port to forward from the container's own address to the loopback HTTP listener. Unset means no forwarder. |
| `SB_HTTP_PORT` | `8787` | Where that forwarder points — match it to `http_port` if you changed it. |
| `PYTHONUSERBASE` | `/data/python` | Where runtime `pip install`s land. Change it and store dependencies stop persisting. |

Build arguments: `PYTHON_VERSION` (default `3.14`), `UID` and `GID` (default
`1000`).

---

## Recipes

### A REPL

```bash
docker run --rm -it --init -v sb-data:/data second-brain
```

`-it` is not optional. The REPL frontend claims the container's stdin; without
a terminal attached it reads EOF, closes the console and stops itself, leaving
the app running with no frontend and nothing on screen to explain it.

### Headless, in the background

```bash
docker run -d --name sb --init -v sb-data:/data second-brain
docker logs -f sb
```

With no terminal the REPL bows out, which is what you want here: the frontends
that matter are the ones you configured — Telegram, HTTP — and they need no
console. It is noisy about it, though, so the entrypoint says what is coming
before it happens: five `frontend repl poll failed` warnings and two errors,
after which the app runs on with no frontend. Nothing is wrong.

`docker stop sb` sends SIGTERM, which the app handles; it exits 0 in well under
a second rather than being killed at the ten-second mark.

### Reaching the web UI

The kernel binds its HTTP server to `127.0.0.1` deliberately: putting an
assistant on a public interface is a decision about exposure, and it belongs to
whoever runs the tunnel. Inside a container, *you* are that operator — and
loopback inside the container is not reachable from `-p` at all. So the image
ships a one-line forwarder, opt-in:

```bash
docker run -d --name sb --init \
  -v sb-data:/data \
  -e SB_EXPOSE_HTTP=8787 \
  -p 127.0.0.1:8787:8787 \
  second-brain
```

`SB_EXPOSE_HTTP` starts the forwarder inside the container; `-p` publishes it
to your machine. You need both — and if only one is a mystery later, the
symptom is identical either way: a port that accepts nothing.

The forwarder listens on the container's *own* address rather than `0.0.0.0`,
which is why it can use the same port number as the app. A wildcard listener on
8787 owns loopback 8787 too, so the app's own bind would fail with `EADDRINUSE`
— reported by the frontend, several layers from the forwarder that caused it.

Then, inside the app: `/packages install frontend_http`, set a long
`secret_http_token` in `/config`, and `/restart`. Point the UI's `VITE_SB_URL`
at `http://localhost:8787`, and set `http_allowed_origins` to the origin the UI
is served from, or the browser blocks the preflight and tells you very little
about why.

Publishing to `127.0.0.1:8787` keeps this on your own machine. `-p 8787:8787`
puts it on your LAN, where the bearer token is the only thing between the
network and your files.

### The test suite

```bash
docker run --rm --init second-brain test
docker run --rm --init second-brain test tests/test_sandbox_policy.py -n0
```

Arguments after `test` go straight to pytest. No volume needed — the suite
builds its own temporary trees, and a container with an empty `/data` is a
clean machine, which is exactly what makes this worth doing. The kernel is
written to be OS-agnostic and mostly developed on Windows; the first Linux run
found four things.

It is also the *whole* suite, which a Linux run without `.git` is not: the
store-dependent tests reach a real `git ls-tree`, so an image built without the
repository skips 83 of them and reports a clean run. 2593 passed, 9 skipped.

### One-off commands

Anything the entrypoint does not recognise is executed as given:

```bash
docker run --rm second-brain python -c "import parsing, llm; print(parsing.discover(), llm.discover())"
docker run --rm second-brain git log --oneline -5 origin/store
docker run --rm -v sb-data:/data second-brain pip list
docker run --rm -it -v sb-data:/data second-brain shell
```

### A preconfigured container

Anything at `/seed` is copied into the data tree the first time it comes up
empty — first boot only, decided by whether `config.json` exists, so it can
never overwrite a tree somebody has been living in.

Mount one:

```bash
docker run --rm -it --init -v $PWD/my-template:/seed:ro -v sb-data:/data second-brain
```

Or bake one into your own image:

```dockerfile
FROM second-brain:latest
COPY --chown=1000:1000 my-template/ /seed/
```

A template is just a data tree: `config.json`, `plugin_config.json`, and
whatever `installed/` packages you want present from the start.

### A benchmark container

The seed mechanism plus `--rm` is the whole thing. Every run starts from the
same known state and leaves nothing behind, because no volume means the data
tree is the container's own writable layer:

```dockerfile
FROM second-brain:latest
COPY --chown=1000:1000 golden-template/ /seed/
```

```bash
docker run --rm --init \
  --network none \
  --memory 2g --cpus 2 --pids-limit 512 \
  -e SB_SEED_DIR=/seed \
  second-brain-bench
```

`--network none` is worth stating explicitly for a benchmark that should not
reach the internet — the store fetch degrades silently to the local ref, which
is the behaviour you want. The resource limits matter more here than elsewhere:
the thing under test starts processes.

Note that a golden template freezes the *store commit* it was built from, since
`installed/` holds copies of files rather than references. That is a feature
for reproducibility and a trap for freshness; rebuild the template when the
store moves.

---

## When it does not work

| Symptom | What it is |
|---|---|
| `exec /app/docker-entrypoint.sh: no such file or directory` | The script has CRLF line endings. `.gitattributes` pins it to LF; check your checkout. |
| Container starts, nothing on screen, exits on any keypress | No `-it`. The REPL got EOF on stdin. |
| `/data is not writable by uid 1000` | A host directory owned by someone else. Rebuild with `--build-arg UID=$(id -u)`, or `chown` it. |
| Published port refuses connections | `SB_EXPOSE_HTTP` is not set, so nothing forwards the loopback listener. |
| `/packages` finds nothing to install | `.git` missing from the image, or the store ref was never fetched. `docker run --rm second-brain git branch -r` should list `origin/store`. |
| A package installs, then fails to load | Its pip dependency went somewhere that did not persist. Check `PYTHONUSERBASE` is `/data/python` and that `/data` is a volume. |
| Everything is gone after a restart | No `-v`. The container's writable layer is not storage. |
| Scheduled jobs fire at the wrong hour | `TZ` is UTC by default. |
| A store package has no wheel for this Python | `docker build --build-arg PYTHON_VERSION=3.13 -t second-brain .` |
