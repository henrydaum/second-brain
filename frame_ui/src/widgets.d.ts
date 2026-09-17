export type WidgetEntry = { name: string; stem: string; tree: string; path: string; extension: string; shadowed?: boolean };
export function listWidgets(): Promise<WidgetEntry[]>;
export function readWidget(widget: WidgetEntry): Promise<string>;
