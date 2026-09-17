/** Insert literal host paths; JSON lists get a properly escaped new item. */
export function insertFormPath(value: string, path: string, mode: string, start: number, end: number): string {
  if (mode === "json") {
    if (!value.trim()) return JSON.stringify([path], null, 2);
    try {
      const parsed: unknown = JSON.parse(value);
      if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) {
        return JSON.stringify([...parsed, path], null, 2);
      }
    } catch { /* Partial JSON remains editable; insert an escaped string at the selection. */ }
    path = JSON.stringify(path);
  }
  return value.slice(0, start) + path + value.slice(end);
}
