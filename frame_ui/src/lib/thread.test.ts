import { describe, expect, it, vi } from "vitest";

import { selectThread, THREAD_STORAGE_KEY } from "@/lib/thread";

function memoryStorage(initial?: string) {
  const values = new Map<string, string>();
  if (initial) values.set(THREAD_STORAGE_KEY, initial);
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    value: () => values.get(THREAD_STORAGE_KEY),
  };
}

describe("browser thread selection", () => {
  it("gives an explicit URL thread precedence without changing storage", () => {
    const storage = memoryStorage("web-saved");
    const randomUUID = vi.fn(() => "unused");

    expect(
      selectThread({
        search: "?thread=windows",
        configured: "development",
        storage,
        randomUUID,
      }),
    ).toBe("windows");
    expect(storage.value()).toBe("web-saved");
    expect(randomUUID).not.toHaveBeenCalled();
  });

  it("keeps an explicitly configured development thread", () => {
    const storage = memoryStorage("web-saved");

    expect(
      selectThread({
        search: "",
        configured: "main",
        storage,
        randomUUID: () => "unused",
      }),
    ).toBe("main");
  });

  it("reuses the browser-local thread across page loads", () => {
    const storage = memoryStorage();
    const first = selectThread({
      search: "",
      storage,
      randomUUID: () => "device-one",
    });
    const second = selectThread({
      search: "",
      storage,
      randomUUID: () => "device-two",
    });

    expect(first).toBe("web-device-one");
    expect(second).toBe(first);
    expect(storage.value()).toBe(first);
  });

  it("still produces a page-lifetime identity when storage is unavailable", () => {
    const storage = {
      getItem: () => {
        throw new Error("disabled");
      },
      setItem: () => {
        throw new Error("disabled");
      },
    };

    expect(
      selectThread({
        search: "",
        storage,
        randomUUID: () => "private-window",
      }),
    ).toBe("web-private-window");
  });
});
