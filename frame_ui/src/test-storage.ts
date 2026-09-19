/**
 * A working `localStorage` for every test, whatever Node supplies.
 *
 * Node 25 defines a global `localStorage` of its own, and without a valid
 * `--localstorage-file` it is an **empty object** rather than a Storage — no
 * `clear`, no `getItem`, nothing. Vitest's jsdom environment does not replace a
 * global that already exists, so `window.localStorage` resolves to that stub
 * and every `localStorage.clear()` in an `afterEach` throws.
 *
 * That failure is worse than it looks: a throwing `afterEach` abandons the rest
 * of the teardown, so module-level fixtures leak into the next test and files
 * fail with assertions about state they never set — which is how one missing
 * method presented as nine unrelated panel failures and three binding ones.
 *
 * So the check is for a usable Storage rather than for a Node version: a real
 * one is left alone, anything else is replaced with an in-memory equivalent.
 */
function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() { return values.size; },
    key: index => [...values.keys()][index] ?? null,
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, String(value)); },
    removeItem: key => { values.delete(key); },
    clear: () => { values.clear(); },
  } as Storage;
}

function usable(candidate: unknown): boolean {
  const storage = candidate as Storage | undefined;
  return typeof storage?.clear === "function"
    && typeof storage?.getItem === "function"
    && typeof storage?.setItem === "function";
}

if (!usable(globalThis.localStorage)) {
  const storage = memoryStorage();
  // Both lookup paths, because component code says `localStorage` and tests
  // reach for `window.localStorage`; one of the two being the stub is the
  // same bug in a less obvious place.
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true, value: storage,
  });
  if (typeof window !== "undefined") {
    Object.defineProperty(window, "localStorage", {
      configurable: true, value: storage,
    });
  }
}
