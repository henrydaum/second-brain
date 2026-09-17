import { useEffect, useRef } from "react";
import { readMotion } from "@/lib/motion";

/** Animate replacement content in place: no remount, stale interactive copy,
 * or delay before the next screen can accept input. */
export function useSurfaceReveal<T extends HTMLElement>(identity: unknown) {
  const ref = useRef<T>(null);
  const interruptedOpacity = useRef<number | null>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element?.animate) return;
    const motion = readMotion(element, "reveal");
    if (motion.duration === 0) {
      interruptedOpacity.current = null;
      return;
    }
    const animation = element.animate(
      [{ opacity: interruptedOpacity.current ?? 0.8 }, { opacity: 1 }],
      motion,
    );
    interruptedOpacity.current = null;
    return () => {
      interruptedOpacity.current = animation.playState === "running"
        ? Number(getComputedStyle(element).opacity) : null;
      animation.cancel();
    };
  }, [identity]);
  return ref;
}
