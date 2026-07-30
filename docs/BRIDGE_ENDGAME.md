# Should the registries speak sandbox natively?

A scoping note, not a plan. Read-only: nothing here was changed.

## The question

`sandbox/bridge.py` is 1,179 lines. Only about 60 of it is the temporary
dual-mode shim that gets deleted when the last plugin is migrated
(`SANDBOX_MODULES`, `is_sandboxed`, `imports_sdk`, `family_of`, the
`if not is_sandboxed(path): return None` fall-through, and the module
docstring's dual-mode argument). The other ~1,100 is adapter code that exists
because `ToolRegistry`, `Orchestrator`, `ServiceRegistry`, `FrontendManager`
and the command registry all expect native `plugins.Base*` subclasses.

So: is that 1,100 lines permanent, or is it work that disappears if the
registries learn to accept a sandboxed plugin directly?

## The finding that changes the answer

**The kernel does not type-check plugins. There is exactly one `issubclass`
against a plugin base in the entire tree** — `plugins/plugin_discovery.py:754`,
and it is not an interface gate. It answers "which class in this module is the
plugin?", which is a *discovery* question, not a calling-convention one.

Everything downstream is duck-typed attribute access. What the registries
actually read off a plugin instance:

| Consumer | Attributes read |
|---|---|
| `agent/tool_registry.py` | `name`, `run`, `to_schema`, `requires_services`, `background_safe`, `max_calls` |
| `pipeline/orchestrator.py` | `name`, `run`, `run_event`, `setup`, `teardown`, `reads`, `writes`, `modalities`, `trigger`, `trigger_channels`, `output_schema`, `batch_size`, `max_workers`, `timeout`, `require_all_inputs`, `requires_services`, `default_jobs` |
| `command_registry.py` | `name`, `description`, `category`, `form`, `require_approval`, `approval_actions`, `approval_prompt`, `approval_actor_id`, `hide_from_help` |

That is the real contract, and it is a **bag of attributes plus two or three
callables** — not a class hierarchy. The native base classes are serving as
(a) a discovery marker and (b) a place for default attribute values. Neither
requires inheritance.

This is much thinner coupling than "1,100 lines of adapter" suggests, and it
means the honest answer to the original question is **neither of the two
options as posed**.

## What the 1,100 lines actually are

Measured, largest first:

| Piece | Lines | Would it survive native registries? |
|---|---|---|
| `_adapt_frontend` | 287 | **Mostly yes.** ~200 of it is the poll loop, the nine-`render_*`-to-one-`render` fan-out, `capabilities` rebuilding and lifecycle inversion. That is *translation between two different shapes*, not subclass plumbing, and something has to do it wherever it lives. |
| `adapt` | 217 | ~120 survives: declaration copying, grant/approval-prompt rendering, box naming, entry resolution. ~90 is subclass construction. |
| `_adapt_service` | 178 | ~120 survives: `exports` forwarding, residency lifecycle, hook/event wiring. |
| poll driving, hook/event wiring, `_form_step`, `_capabilities`, `_cached_prompt` | ~150 | **Yes, all of it.** Pure shape translation. |
| `_result_to_native`, `_build`, `NATIVE_BASES` | ~60 | No — this is the subclassing itself. |

**Realistic saving from native registries: ~250–350 lines, not ~1,100.** The
adapter is mostly translating *shapes* (a guest dict into a `FormStep`, nine
render methods into one call, a return value into a `ToolResult`), and those
shapes differ because the two sides genuinely differ — not because of
inheritance.

## What it would cost

Each registry needs to stop assuming defaults come from a base class. Today
`getattr(tool, "background_safe", True)` works because `BaseTool` sets it; a
sandboxed plugin's declarations are read by AST and may simply be absent. So
every one of the ~35 attributes above needs a defined default *somewhere the
kernel owns* — which is a table, and that table is the contract the base
classes are currently standing in for. Writing it down is arguably the real
deliverable, and it is worth doing whether or not the registries change.

Then: `plugins/plugin_discovery.py:754` needs a second way to find the plugin
class (the validator already computes this — `report.declarations` and
`_plugin_classes` — so it is a rewire, not new logic).

## What survives regardless

- **`BaseFrontend` (916 lines).** Host-side routing: fourteen bus
  subscriptions funnelling into nine `render_*` methods, and `submit_*` into
  `runtime.handle_action`. It is the kernel's, not a plugin's, and
  `_adapt_frontend` subclasses it precisely because that routing should not be
  reimplemented. Keep it under any option.
- **The other four bases**, until the store migration finishes: 39 store
  plugins still subclass them, against 15 migrated. That ratio is the real
  gate on this decision.

## Recommendation

**Do not do this as a project.** Do the useful half of it instead:

1. **Write down the attribute-default table** the base classes currently
   imply. It is needed either way, it is the thing most likely to bite during
   the remaining store migration, and it makes the coupling explicit instead
   of inherited.
2. **Finish the store migration** (39 files). The bridge cannot lose its
   dual-mode half until then, and at 15/54 migrated the endgame is not the
   binding constraint on anything.
3. **Revisit only if `_adapt_frontend` needs real work anyway.** It is the one
   piece large enough to be worth restructuring rather than porting, and the
   two `BaseFrontend` hooks not on the wire (`render_queued_ack`,
   `render_conversation_banner`) are the likeliest reason to reopen it.

The framing to drop is "the bridge is temporary scaffolding". It is not. It is
a **shape translator** between a kernel that thinks in objects and a guest that
can only send data, and that translation is inherent to the sandbox boundary.
Roughly 60 lines of it are scaffolding; the rest is the boundary itself, and
renaming the file when the shim goes would say so.
