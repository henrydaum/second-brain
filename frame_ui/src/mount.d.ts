export type WidgetScheme = "light" | "dark";
export type Widget = { name: string; [key: string]: unknown };
export type WidgetMount = {
  widget: Widget;
  setScheme(scheme: WidgetScheme): void;
  unmount(): void;
};
export function widgetDocument(html: string, options: { token: string; scheme?: WidgetScheme }): string;
export function mountWidget(slot: HTMLElement, widget: Widget, options: { html: string; scheme?: WidgetScheme }): WidgetMount;
