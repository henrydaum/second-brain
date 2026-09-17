import { useSyncExternalStore } from "react";

/**
 * Host scratch paths for attachments that are still in the composer.
 *
 * The upload adapter learns the path, while the attachment tile owns the View
 * affordance. Keeping this transport detail in a tiny shared registry lets the
 * tile use the normal host-file renderer without putting a filesystem path in
 * assistant-ui's message content.
 */
const hostPaths = new Map<string, string>();
const listeners = new Set<() => void>();

const notify = () => listeners.forEach((listener) => listener());

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

export const rememberStagedPath = (id: string, path: string) => {
  hostPaths.set(id, path);
  notify();
};

export const stagedPath = (id: string): string | undefined => hostPaths.get(id);

/**
 * The path as React state.
 *
 * Image tiles could always open from their local object URL. Other files have
 * no local renderer and become viewable only when upload returns a host path,
 * so their tile has to subscribe to that moment rather than merely read the
 * map during its first render.
 */
export const useStagedPath = (id: string): string | undefined =>
  useSyncExternalStore(
    subscribe,
    () => hostPaths.get(id),
    () => undefined,
  );

export const forgetStagedPath = (id: string) => {
  hostPaths.delete(id);
  notify();
};
