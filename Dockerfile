# The kernel in a box. The image holds the app tree and its three kernel
# dependencies; everything heavier installs from the store into DATA_DIR,
# which lives outside the image — bind-mount one, or bake a golden template
# into a derived image (the benchmark path).
FROM python:3.14-slim

# git is not a convenience: the package manager reads the store over it
# (``store_backend.list_tree_files`` shells out to ``git ls-tree``), so a
# container without it can boot but can never install a package — and the
# failure surfaces as a bare FileNotFoundError two layers down.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# paths.py resolves DATA_DIR from XDG_DATA_HOME on Linux, so /data/Second Brain
# is where config, the database and the plugin trees land. There is no
# first-class DATA_DIR override; this env var is the supported knob.
ENV XDG_DATA_HOME=/data
RUN mkdir -p /data

# Headless by default. A frontend is decided by config, not by this file:
# a REPL container needs a TTY (docker run -it), a benchmark container sets
# enabled_frontends to ["http"] in its baked template.
CMD ["python", "main.py"]
