/** @vitest-environment jsdom */
import { afterEach, describe, expect, it, vi } from "vitest";
import { appDocument, attachAppRelay } from "./html-app";
import { RequestFailed, sdk } from "./client";

vi.mock("./client", async importOriginal => ({
  ...await importOriginal<typeof import("./client")>(), sdk: vi.fn(),
}));

afterEach(() => { vi.resetAllMocks(); document.body.replaceChildren(); });

function setup(widgetName?: string) {
  const frame = document.createElement("iframe");
  document.body.append(frame);
  const reply = vi.spyOn(frame.contentWindow!, "postMessage");
  const close = attachAppRelay(frame, "test-token", widgetName);
  const send = (data = {}, source: MessageEventSource | null = frame.contentWindow) => {
    window.dispatchEvent(new MessageEvent("message", { origin: "null", source, data: {
      channel: "second-brain-html-v1", token: "test-token", kind: "call",
      id: 1, type: "conv.list", args: {}, ...data,
    } }));
  };
  return { frame, reply, close, send };
}

describe("HTML App relay", () => {
  it("saves under the originating widget name", async () => {
    vi.mocked(sdk).mockResolvedValue(null);
    const { send, close } = setup("clock");
    try {
      send({ type: "widget.state_set", args: { name: "other", value: { tz: "UTC" } } });
      expect(sdk).toHaveBeenCalledWith("widget.state_set", { name: "clock", value: { tz: "UTC" } });
    } finally { close(); }
  });

  it("installs the helper before App scripts and preserves the page", () => {
    const html = appDocument('<html><head><script>brain.call("conv.list")</script></head><body><button>Go</button></body></html>', "token");
    expect(html.indexOf("window.brain")).toBeLessThan(html.indexOf('brain.call("conv.list")'));
    expect(html).toContain("<button>Go</button>");
  });

  it("forwards arguments and returns results to the originating preview", async () => {
    vi.mocked(sdk).mockResolvedValue([{ id: 4 }]);
    const { send, reply, close } = setup();
    try {
      send({ args: { limit: 3 } });
      await vi.waitFor(() => expect(reply).toHaveBeenCalledWith(expect.objectContaining({
        id: 1, kind: "result", data: [{ id: 4 }],
      }), "*"));
      expect(sdk).toHaveBeenCalledWith("conv.list", { limit: 3 });
    } finally { close(); }
  });

  it("ignores other windows and tokens and rejects unsafe routing and frontend controls", () => {
    const { send, reply, close } = setup();
    try {
      send({}, window);
      send({ token: "another-preview" });
      expect(reply).not.toHaveBeenCalled();
      send({ type: "../events" });
      send({ type: "conv.list", args: [] });
      send({ type: "frontend.resolve" });
      expect(reply).toHaveBeenCalledTimes(3);
      expect(sdk).not.toHaveBeenCalled();
    } finally { close(); }
  });

  it("preserves kernel error details", async () => {
    vi.mocked(sdk).mockRejectedValue(new RequestFailed("fs.read", 403, "approval_declined", "Declined"));
    const { send, reply, close } = setup();
    try {
      send({ type: "fs.read" });
      await vi.waitFor(() => expect(reply).toHaveBeenCalledWith(expect.objectContaining({
        error: { message: "Declined", type: "fs.read", status: 403, code: "approval_declined" },
      }), "*"));
    } finally { close(); }
  });

  it("stops accepting calls and drops late results after closing", async () => {
    let finish!: (value: unknown) => void;
    vi.mocked(sdk).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    const { send, reply, close } = setup();
    send();
    close();
    send({ id: 2 });
    finish("late");
    await Promise.resolve();
    expect(sdk).toHaveBeenCalledTimes(1);
    expect(reply).not.toHaveBeenCalled();
  });
});
