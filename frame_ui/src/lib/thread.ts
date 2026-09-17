/** The browser-local identity used when neither the URL nor development
 * configuration deliberately names a session. Separate browser profiles have
 * separate storage, so a Mac, phone, and PC do not fight over one SSE stream. */
export const THREAD_STORAGE_KEY = "second-brain:thread";

type ThreadStorage = Pick<Storage, "getItem" | "setItem">;

type ThreadSelection = {
  search: string;
  configured?: string;
  storage: ThreadStorage | null;
  randomUUID: () => string;
};

/**
 * Select the session for this browser.
 *
 * An explicit URL remains the escape hatch for a second tab or a deliberately
 * shared session. Development configuration comes next. Production normally
 * reaches the browser-local branch, whose value survives reloads and PWA
 * launches without being shared with another device.
 */
export function selectThread({
  search,
  configured,
  storage,
  randomUUID,
}: ThreadSelection): string {
  const explicit = new URLSearchParams(search).get("thread")?.trim();
  if (explicit) return explicit;

  const fromConfig = configured?.trim();
  if (fromConfig) return fromConfig;

  try {
    const stored = storage?.getItem(THREAD_STORAGE_KEY)?.trim();
    if (stored) return stored;
  } catch {
    // Storage can be disabled by browser privacy settings. A page-lifetime
    // identity is still better than sending every such browser to `main`.
  }

  const generated = `web-${randomUUID()}`;
  try {
    storage?.setItem(THREAD_STORAGE_KEY, generated);
  } catch {
    // The generated value remains stable for this module/page lifetime.
  }
  return generated;
}

/** Accessing the localStorage property itself can throw in a locked-down
 * browser, before getItem is ever called. Keep that failure inside the same
 * graceful fallback as storage operation failures. */
export function browserStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
