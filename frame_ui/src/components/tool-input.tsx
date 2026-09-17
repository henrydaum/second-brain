import { useState } from "react";
import { Button } from "@/components/ui/button";

/** Values stay literal; paths, captions, and commands are never interpreted as markdown. */
function Value({ value }: { value: unknown }) {
  if (Array.isArray(value)) {
    if (!value.length) return <code>[]</code>;
    return <ul className="space-y-1 border-l pl-3">{value.map((item, index) =>
      <li key={index}><Value value={item} /></li>)}</ul>;
  }
  if (value !== null && typeof value === "object") {
    return <details className="min-w-0">
      <summary className="cursor-pointer text-muted-foreground">Object · {Object.keys(value).length} fields</summary>
      <Fields value={value as Record<string, unknown>} />
    </details>;
  }
  return <span className="whitespace-pre-wrap break-all">{typeof value === "string" ? value : String(value)}</span>;
}

function Fields({ value }: { value: Record<string, unknown> }) {
  return <dl className="space-y-2">{Object.entries(value).map(([key, item]) =>
    <div key={key} className="grid min-w-0 gap-1 sm:grid-cols-[minmax(5rem,8rem)_minmax(0,1fr)] sm:gap-3">
      <dt className="text-muted-foreground font-medium break-all">{key}</dt>
      <dd className="min-w-0"><Value value={item} /></dd>
    </div>)}</dl>;
}

export function ToolInput({ args, argsText }: { args: Record<string, unknown>; argsText: string }) {
  const [raw, setRaw] = useState(false);
  let parsed: unknown;
  try { parsed = JSON.parse(argsText || JSON.stringify(args)); } catch { parsed = undefined; }
  const structured = parsed !== null && typeof parsed === "object" && !Array.isArray(parsed);
  const json = structured ? JSON.stringify(parsed, null, 2) : argsText;
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-2">
        <p className="text-muted-foreground flex-1 text-xs font-medium">Input</p>
        {structured && <Button variant="ghost" size="xs" aria-pressed={raw}
          onClick={() => setRaw(!raw)}>{raw ? "Fields" : "Raw JSON"}</Button>}
      </div>
      <div className="bg-muted/60 max-h-64 overflow-auto rounded-md p-2.5 text-xs">
        {structured && !raw ? <Fields value={parsed as Record<string, unknown>} /> :
          <pre className="whitespace-pre-wrap break-all">{json}</pre>}
      </div>
    </div>
  );
}
