/**
 * Saying when something happened, in as few characters as will still answer.
 *
 * **A bare clock time is only unambiguous on the day it happened.** "11:43 PM"
 * in a conversation you opened from last month tells you nothing you wanted to
 * know â€” and the same trap springs again a year later, which is why the ladder
 * below has three rungs rather than two: "Aug 1" is exactly as useless as
 * "11:43 PM" once there is more than one August in play.
 *
 * Everything here formats in the reader's own locale. Month names, day/month
 * order and the 12- or 24-hour clock are regional, and hard-coding one
 * arrangement would be wrong for most of the world.
 */

/** Same calendar day *locally* â€” not within 24 hours, and not the same UTC day.
 *  Both of those get it wrong near midnight, which is precisely when the
 *  distinction is being drawn. */
function sameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

/**
 * The short form, for the line under a message.
 *
 * Today is a time, this year is a date, anything older carries its year. No
 * "Yesterday" rung on purpose: it reads well but it is a special case that
 * changes width and meaning overnight, and "Aug 7" is already unambiguous.
 *
 * `now` is a parameter so this is a pure function of two instants â€” which is
 * what makes it testable without freezing the clock.
 *
 * One honest limitation: a message sent today is labelled with a time, and
 * stays labelled that way until something re-renders it. Left open across
 * midnight, yesterday's messages keep yesterday's clock times until the page
 * next paints them. A timer to correct that costs more than the confusion it
 * saves.
 */
export function shortTimestamp(at: Date, now: Date = new Date()): string {
  if (sameDay(at, now)) {
    return at.toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    });
  }

  if (at.getFullYear() === now.getFullYear()) {
    return at.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  return at.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/** The whole thing, for a `title` â€” the answer to whatever the short form left
 *  out. Always complete, so it never needs a rung of its own. */
export function fullTimestamp(at: Date): string {
  return at.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

/**
 * How long something has been going, for a counter ticking as you read it.
 *
 * Not localised, unlike everything above: this is a duration rather than a
 * moment, its units are one letter each, and it is read beside a running
 * indicator rather than as prose. Minutes appear as soon as there are any,
 * because "83s" makes you do the arithmetic that "1m 23s" has already done.
 */
export function elapsedLabel(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}
