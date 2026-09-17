/**
 * The whole app: a status bar and a chat window, inside the provider that owns
 * the connection.
 *
 * The conversations sidebar and the command palette come next; the layout below
 * leaves the left edge free for them. Everything they need — `conv.list`,
 * `command.list` — is an ordinary Request, so neither changes anything here.
 */

import { Suspense, useState, type FC } from "react";

import { ConversationSidebar } from "@/components/conversation-sidebar";
import { ConversationDocumentTitle } from "@/components/conversation-document-title";
import { ErrorBoundary } from "@/components/error-boundary";
import { InputRequestDialog } from "@/components/input-request-dialog";
import { LazyFileViewerDialog } from "@/components/lazy-file-viewer";
import { SessionBar } from "@/components/session-bar";
import { Thread } from "@/components/thread";
import { WidgetPanel } from "@/components/widget-panel";
import {
  FileActivityProvider,
  useFileActivity,
} from "@/runtime/file-activity-provider";
import { FileExplorerProvider } from "@/runtime/file-explorer-provider";
import { SecondBrainProvider } from "@/runtime/provider";
import { useWidgetPanel } from "@/runtime/widget-mode";

export const App: FC = () => {
  /**
   * Whether the conversations drawer is showing on a narrow screen.
   *
   * It lives here because the two components that need it are siblings: below
   * `md` the sidebar is an overlay that starts off-screen, so the control that
   * opens it cannot be inside it. Above `md` the sidebar is an inline rail and
   * this is simply unused.
   */
  const [navOpen, setNavOpen] = useState(false);

  /**
   * Where the widget is, and whether it is out at all.
   *
   * A hook rather than a provider because exactly two things need it and they
   * are siblings: the button in the session bar and the panel at the other end
   * of the row. When a widget is mounted in there and something *else* wants to
   * open it — a tool result, a render frame naming a widget — that is the
   * moment for a provider, and not before.
   */
  const widget = useWidgetPanel();
  const takeover = widget.open && widget.mode === "full";

  return (
    // Outside the provider: a crash while *setting up* the connection is
    // exactly the case a boundary inside it would miss.
    <ErrorBoundary>
      <SecondBrainProvider>
        <ConversationDocumentTitle />
        {/* Inside the runtime provider, because it reads which conversation is
            open and whether a turn is running; outside the layout, because all
            three of the surfaces that draw files — the header button, the
            drawer, and the chip under each reply — are in different branches
            of it. */}
        <FileActivityProvider>
          <FileExplorerProvider>
          {/*
            A row on a desktop and a column on a phone, and the difference is
            the widget: beside the chat on one, above it on the other. The
            sidebar is an overlay below `md` and takes no space in the column,
            so the two arrangements need no branch beyond this class.
          */}
          <div className="flex h-dvh w-full flex-col overflow-hidden pt-[env(safe-area-inset-top)] md:flex-row">
            {/*
              `display: contents`, so this wrapper adds nothing to either
              layout and exists only to carry one attribute. A fullscreen
              widget is a layer over the app rather than a rearrangement of it,
              which is what keeps the conversation from remounting — but a
              layer that merely *covers* the app leaves everything under it
              focusable and readable to a screen reader. `inert` is the half
              that makes covering mean hidden.
            */}
            <div className="contents" inert={takeover}>
              <ConversationSidebar open={navOpen} onOpenChange={setNavOpen} />
              <div className="sb-chat-shell relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
                <SessionBar
                  onOpenNav={() => setNavOpen(true)}
                  widgetOpen={widget.open}
                  onToggleWidget={widget.toggle}
                />
                <main className="flex-1 overflow-hidden">
                  <Thread />
                </main>
              </div>
            </div>
            {/* The far edge, opposite the conversations. Beside the thread it
                takes width rather than covering it, which is what makes it
                usable while reading. */}
            <WidgetPanel
              open={widget.open}
              mode={widget.mode}
              onClose={widget.close}
              onToggleFull={widget.toggleFull}
            />
          </div>

          {/* Directly under the provider, above everything. A blocked question
              belongs to the *session*, not to the thread it happened during —
              Settings raises them too — so it is a sibling of the whole layout
              rather than something nested inside one part of it. */}
          <InputRequestDialog />
          <FileViewerMount />


          </FileExplorerProvider>
        </FileActivityProvider>
      </SecondBrainProvider>
    </ErrorBoundary>
  );
};

/** Keep the viewer and its file parsers out of the initial bundle. */
const FileViewerMount: FC = () => {
  const { viewing } = useFileActivity();
  if (!viewing) return null;
  return (
    <Suspense fallback={null}>
      <LazyFileViewerDialog />
    </Suspense>
  );
};

