import { describe, expect, it, vi } from "vitest";

// `client.ts` reads `window.location` as it loads, to work out which thread
// this browser is. Nothing under test here makes a Request, but importing the
// module is enough to need a stub in a suite that runs without a DOM.
vi.mock("@/lib/client", () => ({ sdk: vi.fn() }));

const { toFileEvents } = await import("@/lib/ledger");
type LedgerRow = import("@/lib/ledger").LedgerRow;

/** A row, with the boring columns filled in. `data` is given as an object and
 *  serialised here, because that is how the kernel stores it. */
function row(
  partial: Partial<Omit<LedgerRow, "data_json">> & { data?: unknown },
): LedgerRow {
  const { data, ...rest } = partial;
  return {
    id: 1,
    ts: 1_700_000_000,
    origin: "sandbox",
    action_type: "fs.write",
    conversation_id: 7,
    ok: 1,
    error_code: null,
    args_json: "{}",
    data_json: JSON.stringify(data ?? {}),
    ...rest,
  };
}

describe("toFileEvents", () => {
  it("reads a write from data_json", () => {
    const events = toFileEvents([
      row({
        data: {
          paths: ["/srv/app/notes.md"],
          bytes: 14400,
          level: "safe",
          reason: "workspace",
        },
      }),
    ]);
    expect(events).toEqual([
      {
        rowId: 1,
        ts: 1_700_000_000_000,
        path: "/srv/app/notes.md",
        effect: "wrote",
        viaShell: false,
        bytes: 14400,
      },
    ]);
  });

  it("reads a row of the shape a live kernel actually writes", () => {
    // Copied from a real `ledger.read`, Windows paths and all — the server is
    // a Windows host here, and every path helper splits on both separators
    // because of rows exactly like this one.
    const events = toFileEvents([
      row({
        id: 21910,
        ts: 1786322082.5270715,
        origin: "sandbox",
        action_type: "fs.write",
        conversation_id: 160,
        data: {
          chain: "http:web -> edit_file",
          level: "safe",
          reason: "write in the agent's own tree",
          paths: [
            "C:\\Users\\henry\\AppData\\Local\\Second Brain\\workspace\\scripts\\render_fractal.py",
          ],
          bytes: 6434,
        },
      }),
    ]);
    expect(events).toEqual([
      expect.objectContaining({ effect: "wrote", bytes: 6434 }),
    ]);
  });

  it("converts the ledger's seconds into milliseconds", () => {
    // The ledger speaks fractional epoch seconds and `Turn.createdAt` speaks
    // milliseconds; getting this backwards dates every edit to January 1970.
    const [event] = toFileEvents([
      row({ ts: 1_786_158_850.68, data: { paths: ["/a"] } }),
    ]);
    expect(event.ts).toBeCloseTo(1_786_158_850_680, 0);
  });

  it("ignores args_json entirely", () => {
    // The cap that makes this necessary: past 4000 characters the whole object
    // is replaced by a `{_truncated_chars, head, tail}` wrapper, and the
    // argument that blows the cap is the file's own contents.
    const events = toFileEvents([
      row({
        args_json: JSON.stringify({
          _truncated_chars: 82_000,
          head: '{"path": "/srv/app/enormous.md", "content": "aaaa',
          tail: 'aaa"}',
        }),
        data: { paths: ["/srv/app/enormous.md"] },
      }),
    ]);
    expect(events.map((e) => e.path)).toEqual(["/srv/app/enormous.md"]);
  });

  it("reads a shown file from attachments", () => {
    const events = toFileEvents([
      row({
        origin: "agent_enact",
        action_type: "call_tool",
        data: {
          attachments: ["/srv/app/chart.png", "/srv/app/report.md"],
          llm: "claude-sonnet-5",
        },
      }),
    ]);
    expect(events).toEqual([
      expect.objectContaining({ path: "/srv/app/chart.png", effect: "shown" }),
      expect.objectContaining({ path: "/srv/app/report.md", effect: "shown" }),
    ]);
  });

  it("takes fs.move as source first, destination second", () => {
    const events = toFileEvents([
      row({
        action_type: "fs.move",
        data: { paths: ["/srv/old.md", "/srv/new.md"] },
      }),
    ]);
    expect(events.map((e) => [e.path, e.effect])).toEqual([
      ["/srv/old.md", "moved-from"],
      ["/srv/new.md", "moved-to"],
    ]);
  });

  it("marks an fs.delete as deleted", () => {
    const [event] = toFileEvents([
      row({ action_type: "fs.delete", data: { paths: ["/srv/gone.md"] } }),
    ]);
    expect(event.effect).toBe("deleted");
  });

  it("takes the deleted subset of a shell command, and flags the rest", () => {
    const events = toFileEvents([
      row({
        action_type: "proc.run",
        data: {
          paths: ["/srv/app/build", "/srv/app/dist"],
          deleted: ["/srv/app/build"],
          via: "shell",
          command: "rm -rf build && mkdir dist",
        },
      }),
    ]);
    expect(events.map((e) => [e.path, e.effect, e.viaShell])).toEqual([
      ["/srv/app/build", "deleted", true],
      ["/srv/app/dist", "wrote", true],
    ]);
    expect(events[0].command).toBe("rm -rf build && mkdir dist");
  });

  it("drops rows that failed", () => {
    expect(
      toFileEvents([row({ ok: 0, data: { paths: ["/srv/never.md"] } })]),
    ).toEqual([]);
  });

  it("drops this browser's own uploads, which wear the same origin", () => {
    // Verified against a live ledger: `uploadToHost` writes an attachment to
    // scratch through the same `fs.write_bytes` and is recorded with
    // `origin: "sandbox"`, exactly like a file the agent wrote. Only the
    // chain's last hop tells them apart.
    expect(
      toFileEvents([
        row({
          action_type: "fs.write_bytes",
          origin: "sandbox",
          data: {
            chain: "http:web -> frontend:http",
            reason: "write in scratch",
            paths: ["/workspace/temp/frontend_http-abc.wav"],
          },
        }),
      ]),
    ).toEqual([]);
  });

  it("keeps a write the agent made from the same session", () => {
    const events = toFileEvents([
      row({
        origin: "sandbox",
        data: {
          chain: "http:web -> edit_file",
          paths: ["/workspace/scripts/render.py"],
        },
      }),
    ]);
    expect(events).toHaveLength(1);
  });

  it("keeps a shown file however it got there", () => {
    // The chain test is about edits only — a file the agent chose to show you
    // is not something this browser could have written.
    const events = toFileEvents([
      row({
        data: {
          chain: "http:web -> frontend:http",
          attachments: ["/srv/chart.png"],
        },
      }),
    ]);
    expect(events).toHaveLength(1);
  });

  it("passes over rows that name no paths at all", () => {
    // `npm install`, a glob, a redirect, a subshell — the kernel records no
    // paths rather than guessing, and neither does this.
    expect(
      toFileEvents([
        row({ action_type: "proc.run", data: { command: "npm install" } }),
        row({ action_type: "proc.run", data: {} }),
      ]),
    ).toEqual([]);
  });

  it("survives data_json that is not an object", () => {
    const rows = [
      row({ data_json: "not json at all" } as never),
      row({ data_json: "null" } as never),
      row({ data: { paths: "/srv/not-an-array" } }),
      row({ data: { paths: [42, "/srv/real.md"] } }),
    ];
    expect(toFileEvents(rows).map((e) => e.path)).toEqual(["/srv/real.md"]);
  });

  it("keeps rows in the order they arrived, newest first", () => {
    const events = toFileEvents([
      row({ id: 9, ts: 300, data: { paths: ["/c"] } }),
      row({ id: 8, ts: 200, data: { paths: ["/b"] } }),
      row({ id: 7, ts: 100, data: { paths: ["/a"] } }),
    ]);
    expect(events.map((e) => e.path)).toEqual(["/c", "/b", "/a"]);
  });
});

it("reads all ledger pages, including a poll backlog larger than one page", async () => {
  const { sdk } = await import("@/lib/client");
  const { readLedger } = await import("./ledger");
  vi.mocked(sdk).mockResolvedValueOnce(Array.from({ length: 50 }, (_, i) => row({ id: 100 - i })))
    .mockResolvedValueOnce([row({ id: 50 })]);
  expect(await readLedger(7, 49)).toHaveLength(51);
  expect(sdk).toHaveBeenLastCalledWith("ledger.read", expect.objectContaining({ before_id: 51, since_id: 49 }));
});
