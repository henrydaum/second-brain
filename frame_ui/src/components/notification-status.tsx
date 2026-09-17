import { useEffect, useRef, useState } from "react";
import { afterTransition } from "@/lib/motion";
import { useNotifications } from "@/runtime/provider";
import type { QueuedNotification } from "@/runtime/notifications";

const DISPLAY_MS = 6000;

/** FIFO summaries never cover the transcript or mark the notification read. */
export function NotificationStatus() {
  const { notificationQueue, dismissQueuedNotification, notificationsOpen } = useNotifications();
  const current = notificationQueue.at(-1);
  return (
    <div
      data-slot="notification-status"
      className="sb-notification-line pointer-events-none absolute inset-x-0 bottom-0 z-10 flex h-(--composer-bottom-space) min-w-0 items-center justify-center text-center text-xs leading-4"
      role="status"
      aria-live={notificationsOpen ? "off" : "polite"}
      aria-atomic="true"
      aria-hidden={notificationsOpen || undefined}
      style={notificationsOpen ? { visibility: "hidden" } : undefined}
    >
      {current && <StatusText key={current.key} queuedNotification={current} onExpire={dismissQueuedNotification} />}
    </div>
  );
}

function StatusText({ queuedNotification, onExpire }: { queuedNotification: QueuedNotification; onExpire: (key: string) => void }) {
  const [closing, setClosing] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    const timer = setTimeout(() => setClosing(true), DISPLAY_MS);
    return () => clearTimeout(timer);
  }, []);
  useEffect(() => {
    if (!closing || !ref.current) return;
    return afterTransition(ref.current, "opacity", () => onExpire(queuedNotification.key));
  }, [closing, queuedNotification.key, onExpire]);

  const title = queuedNotification.notification.title.replace(/\s+/g, " ").trim();
  const body = queuedNotification.notification.body.replace(/\s+/g, " ").trim();
  const text = title && body ? `${title} — ${body}` : title || body;
  return <p ref={ref} className="sb-notification-status min-w-0 truncate" data-closing={closing} title={text}>{text}</p>;
}
