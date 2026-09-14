# Performance survey

Date: 2026-09-14. Kernel revision: `37a5b9ad`. Adjacent store worktree: `26b6ff57`.

The strongest opportunities are removing avoidable waits and keeping rendering and background work off the interactive path. Ordinary history-list construction and the minimal conversation loop are already cheap. Start with HTTP wakeups, the subagent barrier, model-pool scheduling, and isolation of slow stream consumers; optimize Python allocations after those.

This is a source survey with offline component measurements, not a production load test. The initial survey made no application changes; the follow-up below implements selected findings. The benchmark uses synthetic data and a temporary database, with no provider calls or live user configuration. The React UI is a separate repository and was not available in this checkout; browser rendering, layout, markdown parsing, and client request waterfalls remain unmeasured.

## Implemented follow-up: easy wins

Implemented at the user's request on 2026-09-14:

- **HTTP wakeups:** request arrival and completed detached frontend actions signal the adapter's poll event. The event is cleared before checking for work so arrivals during a poll are not lost. Waiting occurs outside the frontend box, leaving rendering available. Stop/release also wake the frontend; rejected or stale owners cannot replace a successor's event. The declared polling interval remains a fallback for periodic plugin work.
- **Subagent wakeups:** child completion/cancellation and incoming user input wake the owner's barrier. Cancellation is published before stopping children, so their completion signals cannot restart a cancelled parent. Deadline/fallback checks remain, and per-owner wait events are removed on exit.
- **Streaming filter:** held text accumulates in chunks; ordinary fragments skip tag scans; partial tags use a precomputed prefix set and a bounded suffix check. Existing reasoning/tag semantics are preserved.

The child-finished-after-50-ms probe returned in **64.837 ms**, versus **1,003.158 ms** in the survey baseline. A same-process comparison against the committed filter, with identical synthetic input and three samples per case, measured median **68.579 → 1.706 ms** for 100k tagged characters and **1,699.473 → 15.214 ms** for 1M characters. These measure local synthetic workloads, not provider latency or overall response speedup.

Validation: **297 focused tests passed**, covering HTTP transport, real frontend sockets/streaming, detached action completion, arrival-before-wait races, ownership changes, shutdown, subagents, runtime input/cancellation ordering, and filter behavior. A real-socket regression uses a deliberately long 5-second fallback and requires the response within a 1-second socket timeout, establishing that arrival wakes the poller rather than waiting for its timer.

The broader suite run had **3,013 passing tests, 5 skips, and one unrelated failure**: `tests/test_store_memory_bundle.py::test_the_corpus_is_actions_and_facts_live_elsewhere` expects the literal phrase `there is nothing to write` in store `tools/tool_memory.py`. Both that test and the store source are unchanged, and the failure reproduces when run alone. The final cancellation-ordering change and its two new assertions were covered by the subsequent 297-test focused run.

Remaining findings below describe the survey baseline. Model-pool scheduling, asynchronous socket delivery, database concurrency, and history reconciliation are deliberately left for a broader change; they require more lifecycle/ordering work than this easy-wins pass.

## Scope and existing strengths

Reviewed the paths through runtime/state machine, prompts and hooks, model registry, sandbox execution/IPC, event bus, frontend adapters and HTTP transport, persistence/SQL, attachments/parsing, indexing/watchers, configuration, plugin discovery, and bundled services/commands. Followed the adjacent store's HTTP frontend, Telegram streaming, and LiteLLM backend where they complete the user-facing path. This is not an exhaustive audit of every optional store package.

Existing optimizations worth retaining:

- Streaming model output; one-way backend delta notices; successful `llm.delta` and `http.push` requests excluded from the action ledger.
- Persistent model boxes, warm subprocess provisioning, prompt-cue caching, stable system-prefix placement, and a turn-stable clock.
- SQLite WAL, `synchronous=NORMAL`, foreground-priority locking, paginated conversation reads, and targeted latest-marker retrieval.
- Asynchronous frontend actions so a turn does not occupy the frontend's polling call while waiting for a model or approval.
- Cancellation, provenance, per-user isolation, and crash-recovery checkpoints. Optimizations must preserve these properties.

## Prioritized findings

Effort is relative implementation complexity, not a time estimate. Numeric savings below are either explicitly measured or derived from an interval; they are not promises of production improvement.

| Priority | Opportunity | User-visible effect | Effort / confidence |
|---|---|---|---|
| 1 | Wake HTTP frontend on arrival/completion | Remove idle polling delay from essentially every HTTP interaction | Medium / confirmed wait |
| 1 | Signal subagent barrier | Remove up to roughly 1 s of detection delay on child completion, input, or cancellation | Small–medium / reproduced |
| 1 | Lease the next available model box | Prevent waiting behind an arbitrarily slow call while another box is idle | Medium / confirmed selection behavior |
| 1 | Decouple stream delivery from slow clients | Improve stream smoothness and prevent frontend-wide stalls | Medium–large / confirmed coupling, live magnitude unmeasured |
| 2 | Reduce global database lock occupancy | Improve tail latency during indexing, queries, retention, and long-history writes | Medium–large / confirmed serialization |
| 2 | Stop redundant background transcript replacement | Reduce post-turn work and database contention as histories grow | Medium / measured component cost |
| 2 | Reuse workers for in-process box calls | Reduce thread churn on polling and per-fragment rendering | Medium / measured component cost |
| 2 | Cache stable prompt/config inputs with invalidation | Reduce repeated reads and plugin-boundary calls before model requests | Medium / profile first |
| 3 | Make reasoning buffering linear | Reduce CPU/allocation growth for unusually long tagged streams | Small / measured scaling |
| 3 | Optimize attachment and tool preparation selectively | Shorten attachment/tool-heavy input paths | Medium / workload dependent |

### 1. HTTP polling adds avoidable delay

Evidence: store `frontends/frontend_http.py:192,333` declares a 20 ms poll interval; [`_drive_polls`](../sandbox/residency.py) at lines 191–239 waits after an empty poll. [`HttpServer._accept`](../sandbox/http_server.py) at lines 414–431 queues requests without waking this loop. Completed actions are collected in the next frontend `_deliver` pass.

An arrival at a random phase of an otherwise idle 20 ms polling cycle waits approximately 10 ms on average and nearly 20 ms in the worst phase, before scheduling and sandbox overhead. Completion collection can add another polling wait; it is a separate stage, not an automatic extra 20 ms on every message. Outbound `render` itself is immediate rather than gated by this interval.

Add a host-side work event/condition signaled by request arrival, completed frontend actions, and shutdown. Wait outside `box.call` so the box remains available to render. Use a generation counter or predicate checked under a lock to avoid lost wakeups. Keep a fallback timer for plugins that genuinely need periodic polling. Reducing the interval alone increases idle CPU and thread churn.

Validation: measure request-accepted → action-start and action-completed → response-written, including an arrival racing with entry to the idle wait and a burst of unrelated requests.

### 2. The subagent barrier sleeps for one second

Evidence: [`runtime/subagents.py`](../runtime/subagents.py), `BARRIER_POLL_SECONDS = 1.0` at line 59 and `_barrier` at lines 740–780. It scans finished children and pending user input, then calls `time.sleep`.

The offline probe completed its synthetic child after 50 ms; the actual barrier returned after **1,003 ms**, leaving roughly 953 ms of avoidable waiting. This is conditional on reaching the barrier, not a tax on ordinary turns. The same sleep delays the barrier's observation of new user input and cancellation.

Use a per-owner condition/event signaled by child completion, incoming input, and cancellation, with timeout only for the nearest child deadline. Preserve the rule that new user input outranks passive waiting and that unfinished children remain collectible. Test all wakeup sources and the check-before-wait race.

### 3. Model-pool overflow queues behind box zero

Evidence: [`Brain._lease`](../llm/registry.py) at lines 542–563 returns `self._boxes[0]` whenever the pool is full and no box is idle. Callers then wait on that box's lock. `_release` adds a different completed box to `_idle`, but cannot redirect callers already committed to box zero.

The offline state probe confirms this selection. With two busy boxes, one slow and one fast, an overflow request selects box zero and stays attached to it after box one becomes idle. The resulting delay depends on model duration and overlap; it can greatly exceed local CPU costs. Several interactive sessions and background model use can exceed the pool's assumption of one foreground caller plus subagents.

Use a condition-backed idle queue with exclusive leases; wait for any released box, then lease it. Make lease waiting cancellable and arm the box-specific interrupt only after obtaining an exclusive lease. Handle box death, pool resizing, and load/grow races. Consider foreground priority with bounded fairness rather than simply increasing the pool ceiling and memory use.

Validation: staggered call durations under saturation, queued cancellation, a dead box, concurrent growth, and multiple foreground sessions. Assert that an idle box never coexists with work waiting solely on another box.

### 4. Streaming is synchronously coupled to frontend and socket work

Evidence: [`EventBus.emit`](../events/event_bus.py) calls subscribers on the caller's thread (lines 54–62); [`streams.deliver`](../sandbox/streams.py) invokes the sink directly; native frontend `on_bus_agent_text_delta` routes to render; [`residency._render`](../sandbox/residency.py) calls the frontend box synchronously (lines 794–808). Store HTTP `render` performs `sdk.http.push` for every frame (lines 471–500), ending in `_Response.frame` and `_Handler._write` in [`http_server.py`](../sandbox/http_server.py), which write/flush synchronously.

The backend notice is already one-way: there is no reply round trip to remove there. However, host processing/delivery still waits for rendering, a shared frontend box, and socket output. A slow client can hold the frontend box and delay its other sessions and polls. Pipe buffering may temporarily hide this from the producing backend; it does not remove the delivery bottleneck.

Give transport output bounded per-stream queues and dedicated draining, with explicit slow-consumer handling. Preserve sequence numbers, final/aborted events, replay, attendance, and shutdown ordering. Flush the first visible fragment immediately. Coalesce subsequent fragments only when a backlog exists or within a tightly measured display budget; unconditional batching adds latency.

Do not make the entire event bus asynchronous: approvals, session ordering, and other stateful subscribers rely on its semantics. Decouple the presentation boundary specifically. Instrument subscriber time first. The synthetic slow-subscriber probe demonstrates accumulated synchronous cost, not typical frontend latency.

### 5. WAL does not remove the application-wide database mutex

Evidence: [`Database`](../pipeline/database.py) uses one SQLite connection and `_PriorityLock` (lines 146–198). `query` and `query_rows` hold it across execute/fetch (lines 1194–1244); bulk writes and transcript replacement use the same lock. Foreground priority selects the next waiter; it cannot preempt a long query or write already holding the lock.

Separate read connections can allow WAL's reader/writer concurrency to benefit the application. Start with expensive read-only plugin/search work rather than redesigning every database API. Keep per-connection authorizers, functions, foreign-key and cache settings correct. Add query execution budgets using a connection-local progress handler; a returned-row cap does not bound a sort, aggregate, scan, or recursive query's execution time.

Measure lock wait separately from SQL time. Keep write transactions short and bound indexing/retention batches. Do not remove required transactions or weaken permission checks. Per-connection cache size also matters: copying the existing 50 MB cache budget to many connections would undermine resource efficiency.

### 6. Background turns rewrite messages already saved incrementally

Evidence: [`iterate_agent_turn`](../runtime/persistence.py) replaces the full history after a successful non-queued turn without a compaction checkpoint. [`ConversationLoop._record`](../runtime/conversation_loop.py), lines 1696–1732, already persists each new row. [`replace_conversation_messages`](../pipeline/database.py), lines 1939–1984, deletes and reinserts all message rows under the shared database lock.

With synthetic 1,000-character messages, replacement took **5.52 ms for 1,000 rows** and **68.84 ms for 10,000 rows** in the final run; a single append took **0.034 ms**. Larger real tool payloads or slower storage can change this substantially. Repeated full replacement scales with total history on each turn. This is a background-helper path, not evidence that every foreground turn rewrites its history.

Audit mutations made by hooks/turn completion, then append or reconcile only changes. Retain explicit full replacement for operations that actually rewrite history. Preserve compaction behavior, row identity, message timestamps, authorship, attachments, and crash recovery. The current replacement changes row IDs/timestamps, so eliminating it can also make pagination and replay more stable, provided consumers are tested.

### 7. In-process persistent boxes create a thread for every invocation

Evidence: [`InProcessBox._invoke`](../sandbox/boxes.py), lines 318–370, constructs a new daemon thread and event for each call. Poll and render calls use this machinery. The no-op invocation benchmark measured **0.124 ms median / 0.218 ms p95**, excluding the broader dispatch and SDK path.

An idle 20 ms polling frontend can cause roughly 50 such invocations per second, plus renders. Use a reusable worker owned by the box and a call queue, maintaining serial execution, context boundaries, watchdog accounting, and retirement after a runaway call. Do not let abandoned timed-out workers poison a shared global executor. Benefits are likely more noticeable in scheduling jitter and thread creation rate than in low-load median latency.

### 8. Prompt and configuration reads merit targeted caching

Evidence: `_messages` calls the system-prompt callable per model request; [`build_prompt_sections`](../agent/system_prompt.py) reconstructs semi-stable and dynamic sections; `_account_name` reads the database; `_agent_memory` checks existence, reads all of `MEMORY.md`, and only then truncates it (lines 739–762). [`ConversationRuntime.user_setting`](../runtime/conversation_runtime.py), lines 1303–1309, reads/deserializes the user's entire config for each setting lookup. `refresh_specs` also reconstructs scoped schemas/forms on action paths.

Profile counts and time for these operations before introducing broad caches. Cache stable fragments and user config by user/database identity and revision; invalidate on config, identity, permissions, plugin, and scope changes. Reuse scoped schemas by registry/scope revision. For memory, use bounded reads and a content/version cache that observes edits. Do not freeze security state or live tool visibility across actions. Existing prompt-cue caches should be extended rather than bypassed.

Provider prefix caching is separate from local construction cost. The dynamic block already follows prior history. Measure cached input tokens before moving messages or stripping useful context. Hooks may intentionally add retrieval or model calls: measure each hook's contribution, and move only optional work to a background path.

### 9. Long tagged reasoning accumulates via repeated string copies

Evidence: [`ModelTextFilter.feed`](../runtime/token_stripper.py) appends to the instance's `_buffer` with `+=` while a region is open. This can repeatedly copy the growing immutable string. `_partial_tag_tail` also checks multiple suffixes against every known tag for each fragment.

For 20-character fragments inside one tagged region, the final benchmark measured **5.56 ms at 10k characters**, **55.96 ms at 100k**, and **1,122.65 ms at 1M**. These are total batch CPU/wall times with no provider pacing. The million-character case is a stress workload, not a typical user message or time-to-first-token measurement.

Accumulate held chunks in a list and join only when settling/flushing the region. Add a fast path when text contains no possible tag opener, or precompute proper tag prefixes. Preserve unmatched/misplaced-tag behavior and equivalence of batch and streaming output; the existing token-stripper tests encode meaningful edge cases.

### 10. Attachments and tool preparation: measure before broad changes

[`parse_attachment`](../attachments/parse.py) synchronously attempts text parsing before model-capability routing. Reused files can therefore pay repeated parser work; large or remote specialist formats can dominate submission latency. Native attachments may also not need a text fallback immediately. Consider a bounded parsed-result cache keyed by content, parser/config revision, and access scope, plus capability-aware lazy fallback. Keep attachment acceptance feedback prompt while parsing continues, without reordering the eventual input or losing cancellation.

[`Sandbox.start`](../sandbox/facade.py) calls `_prepare`, which validates source and resolves isolation/dependencies on each one-shot run. Cache validation analysis by exact source digest and validator version, with dependency/isolation invalidation; preserve the digest check binding validation to execution. This is an optimization of repeated trusted analysis, not permission to skip validation. Cold heavy-package imports remain a separate cost, partly addressed by the existing persistent model boxes and subprocess reserve.

## Secondary observations and lower priorities

- Pipeline dispatch polls when idle (`pipeline/orchestrator.py:703–704`). Wake on enqueue/completion for prompt indexing and lower idle work, but prioritize the HTTP and barrier waits first. Batched dependency/status queries can reduce database lock traffic during large indexing bursts.
- HTTP serves one request per connection (`sandbox/http_server.py:141–145`) and routes static assets through the plugin/SDK. Connection reuse and a dedicated static-serving path may improve UI startup and repeated API calls; measure the browser waterfall before changing the HTTP ownership model.
- Telegram deliberately throttles streaming updates and uses a pump sleep (store `frontend_telegram.py:1016–1132`). Preserve transport-specific pacing; inspect first-fragment and final-flush delays separately from edit cadence. Lowering all intervals blindly can backfire.
- HTTP replay storage is bounded by event count per session, not a global byte budget (`frontend_http.py:484–487`). Consider a byte budget and inactive-session expiry while preserving reconnect semantics. Resource impact is workload dependent.
- Compaction runs a model call and can add a visible pause. Measure its incidence and duration. A background summary of an immutable prefix is an option only with revision checks and correct tail merging; do not trade conversational correctness for optimistic scheduling.
- Commands, configuration writes, discovery, startup migrations, and package installation are mainly occasional paths. There is no measured reason here to prioritize their tiny Python operations over interactive queueing. Review command-specific SDK round trips when a measured command is slow.

## Offline measurements and reproduction

Run `python dev/benchmark_latency.py` from the repository root. It imports test doubles from `tests/support.py` and measures actual component implementations. Final successful run: Python 3.14.6 on Windows 11, local temporary SQLite database, warm repetitions. Shared-machine timing varies; no isolated CPU or filesystem-cache control was used.

| Component | Median | Sample p95 |
|---|---:|---:|
| Minimal fake-model loop, no DB/frontend/hooks | 0.059 ms | 0.118 ms |
| Provider message-list assembly, 1,000 rows | 0.055 ms | 0.062 ms |
| Provider message-list assembly, 10,000 rows | 0.597 ms | 0.795 ms |
| Persist one 1,000-character message | 0.034 ms | 0.054 ms |
| Replace 1,000 such messages | 5.519 ms | 14.298 ms |
| Replace 10,000 such messages | 68.842 ms | 83.699 ms |
| In-process box no-op invocation | 0.124 ms | 0.218 ms |
| Filter 100k tagged characters, 20-character fragments | 55.962 ms | 67.720 ms |
| Filter 1M tagged characters, 20-character fragments | 1,122.653 ms | 1,145.845 ms |

The barrier probe returned in 1,003.158 ms for a child completed at approximately 50 ms. The model-pool probe confirmed overflow chooses box zero despite later availability elsewhere. An event-bus stress probe of 100 deltas with a subscriber requesting a 2 ms sleep per event took 250.746 ms median; Windows scheduling makes these sleeps longer than the requested minimum.

Each normal benchmark uses 30 repetitions after warmup; replacement uses 10, no-op invocation 100, filtering and synthetic subscriber delay 3. The reported p95 is nearest-rank on those samples, so the small-sample results are descriptive, not statistically robust production percentiles. The tiny fake turn omits real prompt construction, provider, transport, persistence, and installed plugins; it does not establish end-to-end harness overhead.

## Measurement and implementation sequence

1. Add opt-in monotonic timestamps correlated by session/turn/request: HTTP arrival, action start, prompt ready, model lease acquired, provider dispatch, first raw delta, first visible delta, SSE write, action finish. Add browser receipt and first-paint marks in the UI repository. Record durations and sizes, not message bodies or credentials.
2. Implement and verify HTTP wakeups and barrier wakeups. These remove explicit latency floors without changing model behavior.
3. Correct model leasing and isolate slow stream consumers. Test mixed long/short calls, disconnected/slow clients, cancellation, replay, and multiple sessions.
4. Remove unnecessary history replacement and measure database lock contention under indexing and expensive read workloads. Introduce separate read connections only where measurements justify them.
5. Reuse in-process workers and apply digest/revision-based caches to measured hot spots. Optimize the filter independently with its existing behavioral tests.

Benchmark short text, attachments, command/form/approval actions, rapid follow-up/cancel, long histories, tool-heavy turns, and several simultaneous sessions, with indexing both idle and active. Report p50/p95/p99 input-to-feedback, input-to-first-visible-text, completion latency, stream gap distribution, CPU, thread count, RSS, and database wait time. Separate cold from warm runs and provider time from local time. Set numeric latency targets after this baseline; the current component measurements are insufficient to claim a total percentage speedup.
