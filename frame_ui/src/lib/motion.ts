/** CSS owns motion values. Missing styles or reduced motion mean no animation. */
export function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

export function cssTimeMs(value: string): number {
  const time = value.trim();
  if (!/^-?(?:\d+\.?\d*|\.\d+)(ms|s)$/.test(time)) return 0;
  return parseFloat(time) * (time.endsWith("ms") ? 1 : 1000);
}

export function readMotion(element: HTMLElement, token: "reveal" | "fast") {
  const style = getComputedStyle(element);
  return {
    duration: prefersReducedMotion() ? 0 : Math.max(0, cssTimeMs(style.getPropertyValue(`--sb-motion-${token}`))),
    easing: style.getPropertyValue("--sb-ease").trim() || "linear",
  };
}

/** Follow the selected transition, ignoring child events. The timeout covers
 * missing/cancelled browser events; cleanup prevents callbacks after unmount. */
export function afterTransition(element: HTMLElement, property: string, complete: () => void) {
  const style = getComputedStyle(element);
  const properties = style.transitionProperty.split(",").map((value) => value.trim());
  const durations = style.transitionDuration.split(",").map(cssTimeMs);
  const delays = style.transitionDelay.split(",").map(cssTimeMs);
  let duration = 0;
  properties.forEach((name, index) => {
    if (name === property || name === "all") {
      const active = durations[index % durations.length] ?? 0;
      duration = active > 0 ? Math.max(0, active + (delays[index % delays.length] ?? 0)) : 0;
    }
  });
  if (prefersReducedMotion()) duration = 0;
  let finished = false;
  const cleanup = () => {
    finished = true;
    clearTimeout(timer);
    element.removeEventListener("transitionend", onEnd);
    element.removeEventListener("transitioncancel", onEnd);
  };
  const finish = () => {
    if (finished) return;
    cleanup();
    complete();
  };
  const onEnd = (event: TransitionEvent) => {
    if (event.target === element && event.propertyName === property) finish();
  };
  // A small grace period gives transitionend priority over the fallback.
  const timer = setTimeout(finish, duration > 0 ? duration + 50 : 0);
  element.addEventListener("transitionend", onEnd);
  element.addEventListener("transitioncancel", onEnd);
  return cleanup;
}
