import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * A machine name as a heading.
 *
 * Tool names, command names, config keys and form fields all arrive from the
 * kernel in the same shape — `read_bytes`, `default_llm_profile`,
 * `plugin-watcher` — and all four surfaces that show one to a person want the
 * same thing done to it. Three separate copies of this had drifted only in
 * their name.
 *
 * The empty string comes back empty. A caller that needs a placeholder owns
 * that decision, because what to say instead depends on what is missing.
 */
export function titleCase(value: string): string {
  return value
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
