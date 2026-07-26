# Migrating a plugin to the sandbox

One plugin at a time. The app keeps working throughout — unmigrated plugins
run exactly as they do now, and a migrated one is a single file you can revert
with `git checkout`.

There is no second copy of anything. You rewrite the file in place, and the
harness compares your working tree against the version git still has. The old
version is never registered, so it never collides on `name`.

---

## The loop

### 1. Ask what it involves

```python
from sandbox.migrate import plan
print(plan("plugins/tools/tool_read_file.py").render())
```

```
tool_read_file.py  (tool)
  entry: ReadFile
  signature: run(self, context, **kwargs)  ->  run(self, sdk, **kwargs)

  4 effect(s) to convert:
  line   11: imports 'pathlib', which reaches the environment directly  ->  sdk.fs
  line   14: imports 'paths', which lives on the kernel side of the boundary
  line  104: calls .read_text(), which touches the environment directly  ->  sdk.fs.read

  Requests needed: sdk.fs, sdk.fs.read
```

Every line is something to change and what it becomes. If it says *"mostly a
rename"*, it is.

`plan_tree("plugins/tools")` plans a whole directory, easiest first — start
there when picking what to do next.

### 2. Rewrite the file

Four mechanical changes, then the effects:

| Change | From | To |
|---|---|---|
| Base class | `from plugins.BaseTool import BaseTool` | `from guest.bases import BaseTool` |
| Signature | `run(self, context, **kwargs)` | `run(self, sdk, **kwargs)` |
| Returning | `ToolResult(data=x, llm_summary=s)` | `sdk.ok(x, llm_summary=s)` |
| Failing | `ToolResult.failed(msg)` | `sdk.fail(msg)` |

The signature differs by family, and the **argument order changes** for two of
them:

| Family | Native | Sandboxed |
|---|---|---|
| tool | `run(self, context, **kwargs)` | `run(self, sdk, **kwargs)` |
| task | `run(self, paths, context)` | `run(self, sdk, paths)` |
| command | `run(self, args, context)` | `run(self, sdk, args)` |
| service | `_load(self)` | `start(self, sdk)` + `stop(self, sdk)` |

Then convert each effect the plan listed. The common ones:

```python
open(p).read()              ->  sdk.fs.read(p).data
Path(p).write_text(s)       ->  sdk.fs.write(p, s)
context.db.query(sql, a)    ->  sdk.db.query(sql, a).data
context.services["x"].m()   ->  sdk.services.call("x", "m").data
context.call_tool("t", ...) ->  sdk.tools.call("t", ...)
context.approve_command(...)->  sdk.ui.approve(action, why)
requests.get(url)           ->  sdk.net.http(url)
```

Everything returns a `Result`. Check it:

```python
r = sdk.fs.read(path)
if not r:
    return sdk.fail(r.error)
text = r.data
```

**A denial is an ordinary failure**, not an exception — the user said no, and
your plugin should carry on or report it like any other error.

### 3. Add declarations if it needs them

Only if the defaults are wrong:

```python
box = "gmail"                 # share a process with helper files
isolation = "subprocess"      # this one imports a foreign library
timeout = 120                 # clamped by the kernel; ask freely
requests = ["fs.read"]        # advisory, shown at install time
exports = ["embed"]           # services: what service.call may reach
```

Declaring nothing gets its own in-process ephemeral box, which is right for
most tools.

### 4. Check it still conforms

```python
from sandbox.validator import validate_file
print(validate_file("plugins/tools/tool_read_file.py").render())
```

`conforms.` means it will load. Anything else names the line and the fix.

### 5. Check it still answers the same

```python
from sandbox.parity import compare

v = compare("plugins/tools/tool_read_file.py", "ReadFile",
            payload={"path": "notes.md"}, context=ctx)
print(v.render())
```

```
tool_read_file.py: identical.
```

or

```
tool_read_file.py: 1 difference(s)
  llm_summary:
    native:    'read 40 lines'
    sandboxed: ''
```

Both versions get **the same context object**, so a difference means the
plugin changed, not the world.

### 6. Decide what a difference means

- **Kernel plugins, commands especially** — a difference is a bug until
  proven otherwise. These are load-bearing and users depend on their exact
  output.
- **Store plugins** — a difference is often fine. They were built to be
  customised. Read it, decide, move on.

Only return values are compared. Filesystem and database effects are not, on
purpose: comparing those means two workspaces and a diff, and it is rarely
worth it.

### 7. Commit

The next migration's baseline is this commit. Commit a plugin only when you
are happy with its parity, because `compare` reads `HEAD`.

To compare against something else:

```python
compare(path, "ReadFile", ref="HEAD~3", payload={...})
```

---

## What order to migrate in

Easiest and most provable first. The point of the early ones is to prove the
path, not to test it.

| Tier | What | Why here |
|---|---|---|
| 1 | `read_file`, `glob`, `grep`, `edit_file` | Pure, ephemeral, obvious Requests |
| 2 | Pipeline tasks | Same shape, no session state |
| 3 | Kernel commands | Forms and the command registry |
| 4 | Services, `timekeeper` first | First persistent boxes, `exports` |
| 5 | Store frontends | Needs the inbound protocol |

### What does not get migrated

The sandbox boundary and the kernel boundary are the same boundary. What is
*inside* the kernel is not a plugin on the other side of it:

- **`service_llm`** and **`parser_registry`** — the two plugin modules the
  kernel hard-imports (see CLAUDE.md).
- **`frontend_repl`** — the kernel's own frontend. It *drives* the sandbox;
  it does not run inside it.

That removes the three scariest items from the list, and it is why frontends
can safely be last.

---

## When it goes wrong

**"no HEAD copy of x.py to compare with"** — the file is new or uncommitted.
Nothing to compare; use the validator alone.

**"previous version failed"** — the old plugin could not run with the context
you supplied. Give a fuller context, or compare a simpler payload.

**A difference in `data` only under some inputs** — usually a `Result` vs
`ToolResult` shape mismatch. Check you are returning `sdk.ok(x)` and not
`sdk.ok({"data": x})`.

**Everything is denied** — no approver is wired. `Sandbox(runtime=runtime)`
connects the dialog; without it, everything unsafe is refused by design.
