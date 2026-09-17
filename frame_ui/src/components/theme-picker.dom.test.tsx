/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ThemePicker } from "@/components/theme-picker";
import { THEME_KEY } from "@/lib/theme";

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => {
      values.delete(key);
    },
    setItem: (key, value) => {
      values.set(key, value);
    },
  };
}

beforeEach(() => {
  // Node and jsdom can each install `localStorage`, and some Node/macOS
  // combinations leave a partial object with no `clear`. This test is about
  // theme behaviour, not either runtime's storage implementation, so give both
  // lookup paths one complete deterministic Storage object.
  const storage = memoryStorage();
  vi.stubGlobal("localStorage", storage);
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: storage,
  });
  document.documentElement.classList.remove("dark");
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn().mockReturnValue({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }),
  });
});

afterEach(() => vi.unstubAllGlobals());

describe("ThemePicker", () => {
  it("supports arrow-key selection through the radio menu", async () => {
    const user = userEvent.setup();
    render(<ThemePicker />);

    await user.click(screen.getByRole("button", { name: "Change appearance" }));
    const system = screen.getByRole("menuitemradio", { name: "System" });
    system.focus();
    await user.keyboard("{ArrowDown}{ArrowDown}{Enter}");

    await waitFor(() =>
      expect(window.localStorage.getItem(THEME_KEY)).toBe("dark"),
    );
    expect(document.documentElement).toHaveClass("dark");
  });
});
