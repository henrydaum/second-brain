/**
 * File presentation and disk state have different lifetimes. Live frames own
 * inline placement; only the opening ledger read reconstructs history.
 */
import {
  createContext, use, useCallback, useEffect, useMemo, useRef, useState,
  type PropsWithChildren,
} from "react";
import { readConversationFileTurns } from "@/lib/history";
import { forgetFile } from "@/lib/files";
import { readLedger, toFileEvents, type FileEvent } from "@/lib/ledger";
import { forgetThumbnail } from "@/lib/thumbnails";
import {
  bindByTime, collapse, fileTurns, sameFileTurns, toSections,
  UNATTRIBUTED, type FileEntry, type FileSection,
} from "@/runtime/file-activity";
import { useConversations, useSession } from "@/runtime/provider";
import type { Turn } from "@/runtime/store";

export type Viewing = { paths: string[]; index: number; source?: "explorer" };
export type FileActivity = {
  sections: FileSection[];
  sectionFor: (turnId: string) => FileSection | null;
  recoveredFor: (turnId: string) => string[];
  fileFor: (path: string) => FileEntry | undefined;
  entries: FileEntry[];
  total: number;
  failure: string | null;
  filesOpen: boolean;
  setFilesOpen: (open: boolean) => void;
  openFilesAt: (turnId: string) => void;
  focusTurn: string | null;
  focusRequest: number;
  clearFocus: () => void;
  viewing: Viewing | null;
  view: (paths: string[], index: number, source?: "explorer") => void;
  stepView: (by: number) => void;
  closeView: () => void;
};

export const FileActivityContext = createContext<FileActivity | null>(null);
export function useFileActivity(): FileActivity {
  const value = use(FileActivityContext);
  if (!value) throw new Error("useFileActivity outside FileActivityProvider");
  return value;
}
export function useFileActivityMaybe() { return use(FileActivityContext); }

type LedgerState = {
  conversationId: number | null;
  historical: FileEvent[];
  live: Map<string, FileEvent[]>;
  failure: string | null;
};
const emptyLedger = (conversationId: number | null): LedgerState => ({
  conversationId, historical: [], live: new Map(), failure: null,
});
const EMPTY_PATHS: string[] = [];

/** Latest disk state follows ledger row order, never browser/server clock comparison. */
export function currentFiles(events: FileEvent[], turns: Turn[]): Map<string, FileEntry> {
  const ordered = [...events].sort((a, b) => b.rowId - a.rowId);
  const latest = new Map<string, FileEntry>();
  const collapsed = collapse(ordered);
  const details = new Map([...collapsed.shown, ...collapsed.touched].map((e) => [e.path, e]));
  for (const event of ordered) {
    if (latest.has(event.path)) continue;
    latest.set(event.path, {
      ...details.get(event.path)!,
      path: event.path, ts: event.ts,
      effect: event.effect,
      gone: event.effect === "deleted" || event.effect === "moved-from",
      viaShell: event.viaShell,
    });
  }
  const recorded = new Set(latest.keys());
  for (const turn of turns) {
    for (const part of turn.parts) {
      if (part.kind !== "files" || part.sent) continue;
      for (const path of part.paths) {
        if (!recorded.has(path)) latest.set(path, {
          path, effect: "shown", ts: part.receivedAt ?? turn.createdAt ?? 0,
          gone: false, edits: 1, viaShell: false,
        });
      }
    }
  }
  return latest;
}

export function FileActivityProvider({ children }: PropsWithChildren) {
  const { conversationId } = useConversations();
  const { state } = useSession();
  const [ledger, setLedger] = useState(() => emptyLedger(conversationId));
  const [filesOpen, setFilesOpen] = useState(false);
  const [archive, setArchive] = useState<{ id: number; turns: Turn[]; failure: string | null } | null>(null);
  useEffect(() => {
    setArchive(null);
    if (conversationId === null) return;
    let cancelled = false;
    void readConversationFileTurns(conversationId, () => cancelled).then(
      (turns) => { if (!cancelled) setArchive({ id: conversationId, turns: fileTurns(turns), failure: null }); },
      () => { if (!cancelled) setArchive({ id: conversationId, turns: [], failure: "Conversation file history could not be loaded." }); },
    );
    return () => { cancelled = true; };
  }, [conversationId]);
  const [focusTurn, setFocusTurn] = useState<string | null>(null);
  const [focusRequest, setFocusRequest] = useState(0);
  const [viewing, setViewing] = useState<Viewing | null>(null);
  const turnsRef = useRef(state.turns);
  turnsRef.current = state.turns;
  const selected = useRef(conversationId);
  selected.current = conversationId;
  const pollRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    setLedger(emptyLedger(conversationId));
    setViewing(null);
    setFocusTurn(null);
    if (conversationId === null) return;
    let cancelled = false;
    let cursor = 0;
    let ready = false;
    let busy = false;
    let queued = false;
    const valid = () => !cancelled && selected.current === conversationId;
    const poll = async () => {
      if (!valid()) return;
      if (!ready || busy) { queued = true; return; }
      busy = true;
      // Ownership is captured before awaiting. Shown events from polls are
      // drawer-only: live frames, never a poll's timing, place inline content.
      const last = turnsRef.current.at(-1);
      const owner = last?.role === "assistant" ? last.id : UNATTRIBUTED;
      try {
        const rows = await readLedger(conversationId, cursor);
        if (!valid()) return;
        const fresh = [...new Map(rows.filter((row) => row.id > cursor).map((row) => [row.id, row])).values()];
        cursor = fresh.reduce((max, row) => Math.max(max, row.id), cursor);
        const events = toFileEvents(fresh);
        for (const event of events) {
          forgetFile(event.path);
          forgetThumbnail(event.path);
        }
        setLedger((previous) => {
          if (previous.conversationId !== conversationId) return previous;
          const live = new Map(previous.live);
          for (const event of events) {
            const id = event.effect === "shown" ? UNATTRIBUTED : owner;
            live.set(id, [event, ...(live.get(id) ?? [])]);
          }
          return { ...previous, live, failure: null };
        });
      } catch {
        // Retain known state and retry on the next activity poll.
      } finally {
        busy = false;
        if (queued && valid()) { queued = false; void poll(); }
      }
    };
    pollRef.current = () => { void poll(); };
    void (async () => {
      try {
        const rows = await readLedger(conversationId);
        if (!valid()) return;
        cursor = rows.reduce((max, row) => Math.max(max, row.id), 0);
        setLedger({ ...emptyLedger(conversationId), historical: toFileEvents(rows) });
      } catch {
        if (valid()) setLedger({
          ...emptyLedger(conversationId),
          failure: "File history could not be loaded.",
        });
      } finally {
        ready = true;
        if (valid()) { queued = false; void poll(); }
      }
    })();
    return () => { cancelled = true; pollRef.current = null; };
  }, [conversationId]);

  useEffect(() => {
    pollRef.current?.();
    if (!state.typing) return;
    const timer = window.setInterval(() => pollRef.current?.(), 3000);
    return () => window.clearInterval(timer);
  }, [state.typing, conversationId]);

  const projection = useRef<Turn[]>([]);
  if (!sameFileTurns(projection.current, state.turns)) {
    projection.current = fileTurns(state.turns);
  }
  const projected = projection.current;
  const turns = useMemo(() => {
    const saved = archive?.id === conversationId ? archive.turns : [];
    const current = new Map(projected.map((turn) => [turn.id, turn]));
    // A loaded page can end inside a reply. Keep outputs recovered from its
    // other pages even after that partial reply enters the visible transcript.
    for (const turn of saved) {
      const loaded = current.get(turn.id);
      if (!loaded) { current.set(turn.id, turn); continue; }
      const known = new Set(loaded.parts.flatMap((part) => part.kind === "files" ? part.paths : []));
      const missing = turn.parts.flatMap((part) => {
        if (part.kind !== "files") return [];
        const paths = part.paths.filter((path) => !known.has(path));
        return paths.length ? [{ ...part, paths }] : [];
      });
      if (missing.length) current.set(turn.id, { ...loaded, parts: [...missing, ...loaded.parts] });
    }
    return [...current.values()].sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
  }, [archive, conversationId, projected]);
  const derived = useMemo(() => {
    const held = ledger.conversationId === conversationId ? ledger : emptyLedger(conversationId);
    const historicalTurns = turns.filter((turn) => turn.source !== "live");
    const recoveredBound = bindByTime(held.historical, historicalTurns);
    const bound = new Map(recoveredBound);
    for (const [id, events] of held.live) bound.set(id, [...events, ...(bound.get(id) ?? [])]);
    // Durable identity overrides both legacy clock buckets and poll ownership.
    const owners = new Map(turns.filter((turn) => turn.turnId).map((turn) => [turn.turnId!, turn.id]));
    const tagged: FileEvent[] = [];
    for (const [id, events] of bound) {
      tagged.push(...events.filter((event) => event.turnId));
      bound.set(id, events.filter((event) => !event.turnId));
    }
    for (const event of tagged) {
      const owner = owners.get(event.turnId!) ?? UNATTRIBUTED;
      bound.set(owner, [...(bound.get(owner) ?? []), event]);
    }
    // Retain explicit frame ownership for message counts independently of disk state.
    for (const turn of turns) {
      const shown = turn.parts.flatMap((part) => part.kind === "files" && !part.sent
        ? part.paths.map((path): FileEvent => ({
          rowId: -1, ts: part.receivedAt ?? turn.createdAt ?? 0,
          path, effect: "shown", viaShell: false,
        })) : []);
      if (shown.length) bound.set(turn.id, [...(bound.get(turn.id) ?? []), ...shown]);
    }
    const sections = toSections(bound, turns);
    const allEvents = [...held.historical, ...[...held.live.values()].flat()];
    const files = currentFiles(allEvents, turns);
    const entries = [...files.values()].filter((entry) => !entry.gone)
      .sort((a, b) => a.ts - b.ts || a.path.localeCompare(b.path));
    const byTurn = new Map(sections.map((section) => [section.turnId, section]));
    const recovered = new Map(historicalTurns.map((turn) => [
      turn.id,
      [...new Set((recoveredBound.get(turn.id) ?? [])
        .filter((event) => event.effect === "shown").map((event) => event.path))],
    ]));
    return { sections, files, entries, byTurn, recovered, failure: held.failure ?? (archive?.id === conversationId ? archive.failure : null) };
  }, [ledger, conversationId, turns, archive]);

  const clearFocus = useCallback(() => setFocusTurn(null), []);
  const value = useMemo<FileActivity>(() => ({
    sections: derived.sections,
    sectionFor: (id) => derived.byTurn.get(id) ?? null,
    recoveredFor: (id) => derived.recovered.get(id) ?? EMPTY_PATHS,
    fileFor: (path) => derived.files.get(path),
    entries: derived.entries,
    total: derived.entries.length,
    failure: derived.failure,
    filesOpen, setFilesOpen, focusTurn, focusRequest, clearFocus,
    openFilesAt: (id) => {
      setFocusTurn(id);
      setFocusRequest((request) => request + 1);
      setFilesOpen(true);
    },
    viewing,
    view: (paths, index, source) => {
      const openable = paths.filter((path) => !derived.files.get(path)?.gone);
      const chosen = paths[index];
      if (openable.length) setViewing({ paths: openable, index: Math.max(0, openable.indexOf(chosen)), ...(source ? { source } : {}) });
    },
    stepView: (by) => setViewing((current) => current && ({
      ...current, index: (current.index + by + current.paths.length) % current.paths.length,
    })),
    closeView: () => setViewing(null),
  }), [derived, filesOpen, focusTurn, focusRequest, clearFocus, viewing]);
  return <FileActivityContext value={value}>{children}</FileActivityContext>;
}
