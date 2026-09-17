/**
 * The last resort: show the crash instead of a white page.
 *
 * React unmounts the whole tree when a render throws and nothing catches it,
 * which presents as the window going blank — the single least informative
 * failure a UI can have, because it hides the one thing you need. A boundary
 * turns that into a message and a stack.
 *
 * **Still a class component.** Error boundaries are the one thing hooks cannot
 * express: there is no `useErrorBoundary`, because `componentDidCatch` has no
 * hook equivalent. This is not legacy code to modernise later.
 */

import { Component, type ErrorInfo, type PropsWithChildren } from "react";

type State = { error: Error | null };

export class ErrorBoundary extends Component<PropsWithChildren, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The console still gets it in full — the panel below is a summary, and the
    // component stack is usually what actually names the culprit.
    console.error("second brain: render crashed", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="mx-auto flex h-dvh max-w-2xl flex-col justify-center gap-4 p-8">
        <h1 className="text-lg font-semibold">The interface crashed</h1>
        <p className="text-muted-foreground text-sm">
          The conversation itself is on the server and is unaffected — reloading
          picks it up where it was.
        </p>
        <pre className="border-destructive bg-destructive/10 text-destructive max-h-64 overflow-auto rounded-md border p-3 font-mono text-xs whitespace-pre-wrap">
          {error.stack ?? String(error)}
        </pre>
        <button
          onClick={() => window.location.reload()}
          className="bg-primary text-primary-foreground self-start rounded-md px-3 py-1.5 text-sm"
        >
          Reload
        </button>
      </div>
    );
  }
}
