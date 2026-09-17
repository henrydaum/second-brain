// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { afterTransition, readMotion } from "./motion";

let element: HTMLDivElement;
beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
  element = document.createElement("div");
  element.style.transitionProperty = "opacity, grid-template-rows";
  element.style.transitionDuration = "40ms, 0.2s";
  element.style.transitionDelay = "0s, 10ms";
  document.body.append(element);
});
afterEach(() => { element.remove(); vi.useRealTimers(); vi.unstubAllGlobals(); });

function end(target: HTMLElement, propertyName: string, type = "transitionend") {
  const event = new Event(type, { bubbles: true });
  Object.defineProperty(event, "propertyName", { value: propertyName });
  target.dispatchEvent(event);
}

it("reads CSS seconds, explicit zero, and reduced motion without numeric defaults", () => {
  element.style.setProperty("--sb-motion-reveal", "0.16s");
  element.style.setProperty("--sb-ease", "ease-out");
  expect(readMotion(element, "reveal")).toEqual({ duration: 160, easing: "ease-out" });
  element.style.setProperty("--sb-motion-reveal", "0ms");
  expect(readMotion(element, "reveal").duration).toBe(0);
  element.style.setProperty("--sb-motion-reveal", "100ms");
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  expect(readMotion(element, "reveal").duration).toBe(0);
});

it("ignores descendant and unrelated transitions and completes exactly once", () => {
  const complete = vi.fn();
  afterTransition(element, "grid-template-rows", complete);
  const child = element.appendChild(document.createElement("div"));
  end(child, "grid-template-rows");
  end(element, "opacity");
  expect(complete).not.toHaveBeenCalled();
  end(element, "grid-template-rows");
  vi.runAllTimers();
  expect(complete).toHaveBeenCalledTimes(1);
});

it("uses the computed delay and duration before falling back", () => {
  const complete = vi.fn();
  afterTransition(element, "grid-template-rows", complete);
  vi.advanceTimersByTime(210);
  expect(complete).not.toHaveBeenCalled();
  vi.advanceTimersByTime(50);
  expect(complete).toHaveBeenCalledTimes(1);
});

it("handles cancellation and removes listeners and timers on cleanup", () => {
  const complete = vi.fn();
  const dispose = afterTransition(element, "grid-template-rows", complete);
  dispose();
  end(element, "grid-template-rows");
  vi.runAllTimers();
  expect(complete).not.toHaveBeenCalled();
  afterTransition(element, "grid-template-rows", complete);
  end(element, "grid-template-rows", "transitioncancel");
  vi.runAllTimers();
  expect(complete).toHaveBeenCalledTimes(1);
});

it("does not wait for a transition under reduced motion", () => {
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  const complete = vi.fn();
  afterTransition(element, "grid-template-rows", complete);
  vi.advanceTimersByTime(0);
  expect(complete).toHaveBeenCalledTimes(1);
});
