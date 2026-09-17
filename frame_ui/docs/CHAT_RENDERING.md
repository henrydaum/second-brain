# Reply rendering

Each assistant reply owns ordered text/tool parts, one completed-turn file-outcome group,
one activity line while running, and one final copy/time/file-count footer. Files
are deduplicated by path within that reply; showing a file in a later reply still
creates an outcome there. The drawer independently shows the conversation's latest
known file state, with one row per current path.

Both shared attachments and ledger edits stay drawer-only while the reply is
running. They appear together as a recap after completion, before the footer.
The viewport uses assistant-ui's top anchoring: sending places the newest user
message at the top, and the reply grows below without chasing its bottom.
Tool expansion uses native scroll anchoring rather than a competing scroll lock.

File reconstruction is shared by live and stored turns (`fileTurns`): successful,
answered `show_files` calls contribute their absolute path list from their stored
arguments, in addition to explicit output-attachment records and ledger edits.
Failed/unanswered calls and arbitrary tools' inputs are not outputs. This keeps a
missing attachment ledger row from erasing a confirmed show_files result on reload.
The recap is sorted by path, independent of event arrival order, and only the
footer displays the total. No browser-persisted cache is used to recover recaps.

For other tools or relative show_files inputs, the kernel must persist resolved
output paths (message attachments or ledger attachments). Outputs absent from all
durable records cannot be reconstructed reliably by a frontend.

Live attachment frames establish reply ownership immediately. An attachment frame
after `typing:false` attaches to the last assistant reply without opening another
reply or restarting activity. Tool edits come from ledger polling: ownership is
captured before the request, overlapping polls are serialized, and responses for
an abandoned conversation are ignored. Unattributed ledger rows remain drawer-only.
New kernels persist a nullable `conversation_messages.turn_id`, expose the active
ID through `session.get`, and carry it on stream/tool events and ledger metadata.
The same ID survives queued user messages and subagent barriers/re-drives until
the actual completion, cancellation, or failure. Child agents have their own IDs.
The UI keeps interrupted text in chronological visual segments, but joins their
files and copy text under one final recap/footer. No recap appears merely because
a model text stream ended. Reload and live ledger attribution prefer the durable
ID over timestamps or poll ownership; unknown IDs stay drawer-only until their
messages arrive. Old rows remain nullable with the legacy attribution fallback:
historical turn boundaries are not guessed or backfilled. No new table or browser
persistence is needed. Exact mid-stream text positions are not persisted.

Activity uses visible lifecycle signals:

- **Writing:** an unfinished text stream.
- **Working:** an active tool, with no open text stream.
- **Thinking:** the running interval between those activities.
- **Waiting:** the kernel is blocked at the subagent barrier for child reports.
- **Waiting for your response:** pending user input takes precedence.

The protocol does not expose reasoning tokens or tool-argument generation as
distinct streams. Thinking is therefore an inferred lifecycle label, not a claim
that reasoning tokens are currently arriving. Elapsed intervals reset on activity
changes; completed replies have no activity indicator.

## Validation

Run `npm test`, `npm run build`, and `npm run lint`.

For browser checks, install Chromium with `npx playwright install chromium`, serve
the built app with `npm run preview -- --port 5174`, then run
`node scripts/check-chat.mjs`. This exercises the real application with mocked
backend responses at desktop and mobile widths, including reduced motion, delayed
images, late attachments, gallery expansion, viewer keyboard navigation, repeated
drawer highlights, reload recovery, and reader-aware scrolling. Screenshots go to
the ignored `test-results/chat` directory. No live backend is modified.
