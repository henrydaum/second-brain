import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "@/App";
import "@/index.css";

// StrictMode double-invokes effects in development to surface effects that are
// not safe to run twice. That matters here more than usual: the event stream is
// opened in an effect, and a second `GET /events` on the same thread *replaces*
// the first — so an effect that failed to clean up would silently disconnect
// itself. Leaving StrictMode on means that bug shows up immediately rather than
// in production.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
