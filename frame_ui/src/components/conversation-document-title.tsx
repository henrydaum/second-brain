import { useEffect, type FC } from "react";

import { conversationTitle } from "@/lib/conversations";
import { useConversations } from "@/runtime/provider";

const APP_NAME = "Second Brain";

/** Keep browser chrome aligned with the conversation shown in the thread. */
export const ConversationDocumentTitle: FC = () => {
  const { conversations, conversationId } = useConversations();
  const active = conversations.find(
    (conversation) => conversation.id === conversationId,
  );
  const title = active
    ? `${conversationTitle(active)} - ${APP_NAME}`
    : APP_NAME;

  useEffect(() => {
    document.title = title;
  }, [title]);

  return null;
};
