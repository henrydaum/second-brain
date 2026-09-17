/**
 * What a conversation *is*, editable from the one place it is always named.
 *
 * All of this was previously only reachable through `/conversations`, a form in
 * the settings panel — which meant renaming the thing on screen involved
 * leaving it, finding it in a list, and answering three prompts. The title in
 * the header is where every other chat app puts these, and it is already the
 * thing you point at when you mean "this conversation".
 *
 * **Renaming and filing raise no approval dialog.** `conv.set_title` and
 * `conv.set_category` are both `ALWAYS_SAFE` in the kernel's policy, and each
 * is scoped to the calling user in SQL. Deleting is the unsafe one and behaves
 * differently for it — the kernel asks — but it is here too, because the
 * sidebar's copy of it appears on hover and a phone has no hover to give.
 *
 * **`/conversations` is gone from Settings entirely, and this is what replaced
 * it.** Everything it did, the sidebar and this menu now do in the place the
 * conversation already is — with one exception, traded knowingly.
 *
 * That exception is a conversation's notification mode, which nothing in the UI
 * can now reach. It governs whether a scheduled subagent's result is pushed to
 * you, so it belongs beside the job that does the pushing rather than on every
 * conversation that will never have one — see the scheduled-jobs panel when it
 * exists. Keeping a whole settings page alive as the only door to one setting
 * on the small share of conversations a scheduled job ever writes to was the
 * worse trade; `/conversations` still runs if it is typed.
 *
 * **Clearing and compacting are here for the reason renaming is.** Both act on
 * the conversation on screen and neither is a setting — nothing about them
 * outlives the thread they are pointed at — so a card in a settings dialog was
 * always the wrong shape for them.
 */

import { useEffect, useState, type FC } from "react";
import { ChevronDownIcon, PencilIcon, PlusIcon, TagIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  categoryLabel,
  conversationCategory,
  orderedCategories,
} from "@/lib/conversation-categories";
import { conversationTitle } from "@/lib/conversations";
import { cn } from "@/lib/utils";
import { useConversations, useSession } from "@/runtime/provider";

/** The value a radio item uses for "no category". The empty string is not
 *  usable — Radix treats it as unset — and `null` is not a string. */
const MAIN = " main";

/** Which dialog is up, if either. Both ask for one name, so they share a
 *  component and differ only in what the name is for. */
type Asking = "rename" | "category" | null;

export const ConversationMenu: FC = () => {
  const {
    conversationId,
    openConversationRow,
    conversationCategories,
    renameConversation,
    categoriseConversation,
    deleteConversation,
  } = useConversations();
  const { say, state, status } = useSession();

  const [open, setOpen] = useState(false);
  const [asking, setAsking] = useState<Asking>(null);

  // **From the open conversation's own row, never from the list.** The list is
  // one page of one category, so filing this conversation somewhere the filter
  // excludes takes it out — and looking it up there meant the header lost the
  // conversation the moment you used the header to move it, taking its width
  // with it and collapsing the rest of the row to the left.
  const title = openConversationRow
    ? conversationTitle(openConversationRow)
    : "New chat";
  const category = conversationCategory(openConversationRow?.category);

  // Every category in use, from the server's tally rather than from the loaded
  // page — which is one category's worth of rows and would offer only the
  // category you are already in.
  const categories = orderedCategories(conversationCategories);

  // Nothing to edit until the session is actually pointing at something.
  // `me-auto` here as well as on the trigger: this is what holds the rest of
  // the header against the far edge, and without it the status and the buttons
  // slide over to meet the title.
  if (conversationId === null) {
    return (
      <span
        className="me-auto min-w-0 truncate px-1 text-sm font-medium"
        title={title}
      >
        {title}
      </span>
    );
  }

  return (
    <>
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            // On a phone the title owns all space left between the two control
            // groups. That space changes when Files appears, so a viewport cap
            // would keep reserving room for it after it disappears. Desktop
            // keeps the compact, title-sized control it has always used.
            className="me-auto -mx-1 h-8 min-w-0 flex-1 justify-start gap-1.5 px-2 font-medium md:flex-none md:max-w-[min(28rem,45vw)]"
            aria-label={`Conversation: ${title}`}
          >
            <span className="min-w-0 truncate">{title}</span>
            <ChevronDownIcon className="size-3.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>

        <DropdownMenuContent align="start" className="w-64">
          <DropdownMenuItem onSelect={() => setAsking("rename")}>
            <PencilIcon className="size-4" />
            Rename
          </DropdownMenuItem>

          <DropdownMenuSeparator />

          <DropdownMenuLabel className="text-muted-foreground flex items-center gap-2 text-xs font-normal">
            <TagIcon className="size-3.5" />
            Category
          </DropdownMenuLabel>
          {/* Scrolls past about six. The list is whatever categories this user
              has made, so it has no natural ceiling, and a menu taller than the
              window is one you cannot reach the bottom of. */}
          <div className="max-h-56 overflow-y-auto">
            <DropdownMenuRadioGroup
              value={category ?? MAIN}
              onValueChange={(value) =>
                void categoriseConversation(
                  conversationId,
                  value === MAIN ? null : value,
                )
              }
            >
              <DropdownMenuRadioItem value={MAIN}>
                {categoryLabel(null)}
              </DropdownMenuRadioItem>
              {categories.map((name) => (
                <DropdownMenuRadioItem key={name} value={name}>
                  <span className="min-w-0 truncate">{name}</span>
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </div>

          {/* Outside the scroller, so it stays reachable however many
              categories there are. `conv.set_category` takes any string — a
              category exists because something is filed under it. */}
          <DropdownMenuItem onSelect={() => setAsking("category")}>
            <PlusIcon className="size-4" />
            New category
          </DropdownMenuItem>

          <DropdownMenuSeparator />

          {/**
           * Everything you do *to* a conversation, in one bar across the
           * bottom.
           *
           * **Three across works where two did not, and the count is the
           * reason.** As a pair these sat under a menu whose every label starts
           * at the same inset, close enough to that list to be read as two of
           * its rows and too misaligned to be two of its rows. Three filling
           * the width is not a list at all; it is the footer of one, which is a
           * shape people already know and which owes the rows above it no
           * alignment.
           *
           * **No icons, and that is what makes it fit.** With them the three
           * labels want about 266px inside a menu that has 248px to give, so
           * the row wrapped. The alternative was a wider menu for the sake of
           * three glyphs the words beside them already say.
           *
           * **Left to right is cheapest to dearest, and the colour says so.**
           * Compacting trades the messages for a summary, clearing takes the
           * messages, deleting takes the conversation — yellow, orange, red,
           * the ramp every other warning light uses. It reads in one glance
           * without a label explaining it, and the far edge, where a slip is
           * least likely to land, is the one that costs the most.
           *
           * None of the three is ordinary and only one is destructive, which is
           * what the two colours short of red are for: clearing cannot be
           * undone and compacting leaves a marker nothing removes, so both want
           * to be louder than Rename — and red stops meaning "the conversation
           * is gone" the moment something that keeps it borrows red.
           *
           * Delete earns its place in this menu at all because the sidebar's
           * copy appears on hover, which on a touch screen means it never
           * appears — it was reachable only by someone who already knew.
           *
           * **None of the three confirms, because all three are confirmed
           * already.** `/compact` declares `require_approval`, `/clear` reaches
           * the unsafe `conv.clear`, and `conv.delete` is unsafe outright: the
           * kernel raises its own dialog for each and this waits on it, so
           * asking here would be asking twice.
           *
           * Only the two commands go dim mid-turn, like the model selector's
           * own command item — a command cannot be submitted onto a running
           * turn. Deleting is a Request rather than a command and does not
           * queue behind one.
           */}
          <div className="flex items-center gap-1">
            <DropdownMenuItem
              className="text-notice focus:bg-notice/10 focus:text-notice flex-1 justify-center px-1.5"
              disabled={state.typing}
              onSelect={() => void say("/compact")}
            >
              Compact
            </DropdownMenuItem>
            <Rule />
            <DropdownMenuItem
              className="text-caution focus:bg-caution/10 focus:text-caution flex-1 justify-center px-1.5"
              disabled={state.typing}
              onSelect={() => void say("/clear")}
            >
              Clear
            </DropdownMenuItem>
            <Rule />
            <DropdownMenuItem
              className="text-destructive focus:bg-destructive/10 focus:text-destructive flex-1 justify-center px-1.5"
              disabled={status !== "open"}
              onSelect={() => void deleteConversation(conversationId)}
            >
              Delete
            </DropdownMenuItem>
          </div>

        </DropdownMenuContent>
      </DropdownMenu>

      <NameDialog
        open={asking === "rename"}
        onOpenChange={(next) => setAsking(next ? "rename" : null)}
        heading="Rename conversation"
        description="Second Brain names conversations for you until you name one yourself. After this it will keep the name you give it."
        label="Conversation title"
        initial={title}
        action="Rename"
        onSubmit={(name) => void renameConversation(conversationId, name)}
      />
      <NameDialog
        open={asking === "category"}
        onOpenChange={(next) => setAsking(next ? "category" : null)}
        heading="New category"
        description="Files this conversation under a new name. The category exists as soon as something is in it, and it will appear in the filter beside the others."
        label="Category name"
        initial=""
        action="Create and move"
        onSubmit={(name) => void categoriseConversation(conversationId, name)}
      />
    </>
  );
};

/**
 * A hairline between two cells of the action bar.
 *
 * Decorative, and hidden from the accessibility tree for it: a menu's items are
 * announced as a list already, and a separator between each of three would be
 * three announcements of nothing. `DropdownMenuSeparator` is the horizontal
 * rule that divides *sections* of this menu and carries a real `role`, which is
 * why this is not that.
 */
const Rule: FC = () => (
  <div aria-hidden className="bg-border h-5 w-px shrink-0" />
);

/**
 * Ask for one name, in a dialog rather than inline in the menu.
 *
 * A text field inside a dropdown fights the menu's own keyboard handling —
 * typing moves the selection, Enter picks an item — and Radix is right to do
 * that for a menu. This is the surface that already exists for "one question,
 * two buttons", and both things the menu asks for are that shape.
 */
const NameDialog: FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
  heading: string;
  description: string;
  label: string;
  initial: string;
  action: string;
  onSubmit: (name: string) => void;
}> = ({
  open,
  onOpenChange,
  heading,
  description,
  label,
  initial,
  action,
  onSubmit,
}) => {
  const [draft, setDraft] = useState(initial);

  // Re-seeded each time it opens, not once: the title can have changed since
  // this last opened, and a stale draft would quietly rename the conversation
  // back.
  useEffect(() => {
    if (open) setDraft(initial);
  }, [open, initial]);

  const trimmed = draft.trim();
  const submit = () => {
    if (trimmed && trimmed !== initial) onSubmit(trimmed);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sb-glass sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{heading}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <form
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
          className="flex flex-col gap-4"
        >
          <input
            autoFocus
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            aria-label={label}
            className={cn(
              "border-input bg-background focus-visible:border-ring focus-visible:ring-ring/30",
              "h-10 w-full rounded-lg border px-3 outline-none focus-visible:ring-[3px]",
            )}
          />
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={!trimmed}>
              {action}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};
