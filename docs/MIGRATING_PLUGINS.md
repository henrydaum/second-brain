# Migrating a plugin to the sandbox

For plugins on the **store branch**. Everything in the kernel tree is done.

> **An unmigrated plugin does not load.** It used to: the loader fell through
> to an ordinary import for a file written against `plugins.BaseTool`, which
> is what let the two contracts coexist while the migration ran. That is gone
> — `sandbox.bridge.adapt` is the only way in, and a file it will not carry is
> a capability that silently is not there. Check the log for
> *"did not load: plugins must be written against the SDK"*.

One plugin at a time, rewritten in place. There is no second copy of anything,
and no version of the file is registered but the one you are editing, so it
never collides on `name`.

---

## The loop

### 1. Ask what it involves

```python
from sandbox.validator import validate_file
print(validate_file("plugins/tools/tool_read_file.py").render())
```

```
tool_read_file.py  (tool)
  entry: ReadFile

  4 effect(s) to convert:
  line   11: imports 'pathlib', which reaches the environment directly  ->  sdk.fs
  line   14: imports 'paths', which lives on the kernel side of the boundary
  line  104: calls .read_text(), which touches the environment directly  ->  sdk.fs.read
```

Every line is something to change and what it becomes. If it says *"mostly a
rename"*, it is.

The same call is how you check your work in step 4 — before and after are the
same command, and `conforms.` is the finish line.

### 2. Rewrite the file

Four mechanical changes, then the effects:

| Change | From | To |
|---|---|---|
| Base class | `from plugins.BaseTool import BaseTool` | `from guest.bases import BaseTool` |
| Signature | `run(self, context, **kwargs)` | `run(self, sdk, **kwargs)` |
| Returning | `ToolResult(data=x)` | `return x` |
| Returning with extras | `ToolResult(data=x, llm_summary=s)` | `sdk.ok(x, llm_summary=s)` |
| Failing | `ToolResult.failed(msg)` | `sdk.fail(msg)`, or just let it raise |
| Prompt method | `agent_prompt_for(self, ctx)` | `agent_prompt(self, sdk)` |

The last one is silent if you miss it: the old name is not collected any more,
so the plugin loads fine and contributes nothing to the system prompt.

The signature differs by family, and the **argument order changes** for two of
them:

| Family | Native | Sandboxed |
|---|---|---|
| tool | `run(self, context, **kwargs)` | `run(self, sdk, **kwargs)` |
| task | `run(self, paths, context)` | `run(self, sdk, paths)` |
| command | `run(self, args, context)` | `run(self, sdk, args)` |
| service | `_load(self)` | `start(self, sdk)` + `stop(self, sdk)` |

A command's `form(args, context)` becomes `form(sdk, args)` and is bridged
alongside `run`, so a migrated command keeps collecting its arguments.

For resident services or frontends that need periodic work, move the body of
the old loop or timer into `poll(self, sdk)` and declare `poll_interval`.
Return truthy when more work is already queued (the kernel calls again
immediately) or falsy to wait the interval. Do not create a guest thread or an
`sdk.poll()` request: the kernel owns cadence, serialization, shutdown, and
the repeated-failure limit (`max_poll_failures`, default five).

Then convert each effect the plan listed. The common ones:

```python
open(p).read()               ->  sdk.fs.read(p)
Path(p).write_text(s)        ->  sdk.fs.write(p, s)
context.db.query(sql, a)     ->  sdk.db.query(sql, a)
context.db.conn.execute(...) ->  sdk.db.write(sql, params)
context.services["x"].m()    ->  sdk.services.call("x", "m")
context.call_tool("t", ...)  ->  sdk.tools.call("t", ...)
context.approve_command(...) ->  sdk.ui.approve(action, why)
requests.get(url)            ->  sdk.net.http(url)
logger.info(msg)             ->  sdk.log(msg)
```

**Requests return their value and raise when they fail**, so most plugin code
is a straight line:

```python
def run(self, sdk, path):
    return len(sdk.fs.read(path).split())
```

There is no result to unwrap and no branch that exists only to forward an
error. A failure you do not catch becomes the plugin's failure, carrying the
original reason — which is what the caller wanted anyway.

Catch one only when you have something to do about it. Refusals have their own
class, so "the user said no" can be handled without also swallowing "the disk
is full":

```python
try:
    page = sdk.net.http(url)
except sdk.Denied:
    return "I need permission to fetch that."
```

`sdk.Denied` is a subclass of `sdk.Failed`, so catching `sdk.Failed` catches
both.

### 3. Add declarations if it needs them

Only if the defaults are wrong:

```python
box = "gmail"                 # share a process with helper files
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

Run it. `conforms.` says it will *load*, not that it still does what it did —
that part is yours to check, by calling the plugin and comparing against what
you remember it doing.

Where to look hardest, in order:

- **Kernel plugins, commands especially** — a difference is a bug until proven
  otherwise. These are load-bearing and users depend on their exact output.
- **Store plugins** — a difference is often fine. They were built to be
  customised. Read it, decide, move on.

There was once a `sandbox.parity.compare` that ran the working tree against
`git show HEAD:<path>` and diffed the return values. It is gone, and its
limitation is why: it compared return values only, never filesystem or database
effects, which is where a migration actually goes wrong. A tool that checks the
easy half invites you to skip the hard half.

### 6. Commit

One plugin, one commit — that is what makes a bad migration a `git revert`
rather than an excavation.

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

### Everything gets migrated

Including plugins that drive foreign libraries. A library cannot be reduced to
Requests, so a plugin importing one **loads with a disclaimer** and is run in a
subprocess automatically — the kernel sees the import and decides. That is the
answer the security contract already gives; it is not a reason to leave
anything behind.

One family needs a moment's thought, not an exemption:

- **LLM backends** live in `helpers/llm_*.py` and implement
  `guest.llm.BaseLLMBackend`. The kernel-owned `llm` registry discovers their
  declarations and runs them in boxes; there is no `service_llm` plugin or
  native-backend compatibility path.
- **Console frontends** declare `uses_console = True`, drain input with
  `sdk.console.read_line()` from `poll`, and write with `sdk.console.write()`.
  The kernel owns stdin, so reads never block the box and subprocess stdin
  remains reserved for the wire protocol. If submitting input can render back
  synchronously, declare `background_submit = True`. If the frontend must
  reopen the last conversation before first input, declare
  `restore_on_start = True`; the host restores between box calls so restored
  forms and approvals cannot re-enter a busy box.

---

## When it goes wrong

**"no HEAD copy of x.py to compare with"** — the file is new or uncommitted.
Nothing to compare; use the validator alone.

**"previous version failed"** — the old plugin could not run with the context
you supplied. Give a fuller context, or compare a simpler payload.

**A difference in `data` only under some inputs** — check you are returning
the value itself. `return x`, not `return sdk.ok({"data": x})`.

**Everything is denied** — no approver is wired. `Sandbox(runtime=runtime)`
connects the dialog; without it, everything unsafe is refused by design.
