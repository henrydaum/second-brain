/**
 * Code-split a component, and be able to fetch it early.
 *
 * Four surfaces here are secondary chunks — the file view, the file viewer
 * dialog, the files drawer, Settings — and all four want the same two things:
 * a `lazy` component for the render path, and a bare function the chrome can
 * call on hover, on focus, or a beat after first paint, so that pressing the
 * familiar button is not a first-time download.
 *
 * **There is no module cache here, deliberately.** Each of these used to keep
 * one — `let held = null; held ??= import(…)` — which does nothing: the ES
 * module registry already memoises `import()` by specifier, so every call after
 * the first resolves the promise the first one made, whether it comes from
 * `lazy` or from a preload. Three copies of a cache in front of a cache.
 */

import { lazy, type ComponentType } from "react";

export function lazyWithPreload<Module, Props>(
  load: () => Promise<Module>,
  pick: (module: Module) => ComponentType<Props>,
) {
  const Lazy = lazy(() => load().then((module) => ({ default: pick(module) })));
  const preload = () => void load();
  return [Lazy, preload] as const;
}
