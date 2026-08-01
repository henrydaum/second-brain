# Second Brain — Package Store

This branch (`store`) is the tree-based registry used by the Second Brain
kernel. It is not application code and has no server: the app reads files from
`origin/store` with Git and installs selected entries under
`DATA_DIR/installed`.

## Layout

```text
tools/                  tool_*.py
tasks/                  task_*.py
services/               service_*.py
commands/               command_*.py
frontends/              frontend_*.py
parsers/                parse_*.py
llm/                    llm_*.py
scripts/                SDK code that is run rather than registered
<family>/helpers/       code owned by that family
bundles/                bundle_*.json lists of store-relative files
```

Each Python file is an installable entry identified by its stem. Files may
declare dependencies directly in source:

```python
dependencies_files = ["services/service_example.py"]
dependencies_pip = ["some-package>=1"]
```

`dependencies_files` contains store-relative Python paths. The package manager
installs those first and prunes them on uninstall only when nothing else still
needs them. `dependencies_pip` is authoritative when present; omit it to let
the installer detect third-party imports, or set it to `[]` to install none.

A bundle is a JSON file whose `files` array names store-relative entries. It
groups existing files; it does not contain or replace them.

## Installing

Use `/packages` in Second Brain:

```text
/packages available tools
/packages installed
/packages install tool_web_search
/packages uninstall tool_web_search
```

Installed files retain this tree shape under `DATA_DIR/installed`, where the
kernel discovers them. Helper files are payload rather than entrypoints.

## Authoring and validation

New and migrated plugins use the sandbox SDK (`guest.*`). Validate a source
file with the kernel's validator before publishing:

```python
from sandbox.validator import validate_file
print(validate_file(r"Z:\path\to\tool_example.py").render())
```

From the `main` worktree, validate an entire checked-out store tree with:

```text
python dev/package_publisher.py validate --path "Z:\path\to\store"
```

The validator intentionally permits foreign libraries with a disclaimer;
installed plugins that import them run in subprocess isolation.

## Publishing

The publisher lives on `main` at `dev/package_publisher.py`. It copies files
into a temporary checkout of this branch, validates the complete store, and
commits and pushes the result without modifying the current checkout:

```text
python dev/package_publisher.py publish tool_example --file source/tool_example.py=tools/tool_example.py --require services/service_example.py --pip "some-package>=1"
```

Use `--dry-run` to inspect the generated tree without committing, `--update`
to replace a changed existing file, and `--no-pip` to write an explicit empty
dependency list. Direct pull requests against `store` are also valid; keep the
same layout and run whole-store validation before submitting them.
