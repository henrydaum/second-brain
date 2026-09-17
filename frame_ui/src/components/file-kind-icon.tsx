import { useEffect, useRef, useState, type FC } from "react";
import {
  BookTextIcon,
  FileCode2Icon,
  FileIcon,
  FileTextIcon,
  ImageIcon,
  MusicIcon,
  SheetIcon,
  VideoIcon,
} from "lucide-react";

import { fileUrl } from "@/lib/client";
import { guessIconKind, guessKind, type FileIconKind } from "@/lib/files";
import { thumbnail, thumbnailsWork } from "@/lib/thumbnails";
import { cn } from "@/lib/utils";

const KIND_ICONS: Record<FileIconKind, typeof FileIcon> = {
  code: FileCode2Icon,
  image: ImageIcon,
  video: VideoIcon,
  audio: MusicIcon,
  table: SheetIcon,
  // A note is not a text file with a different extension, and the list is
  // where that shows: a vault is mostly Markdown, so one glyph for both would
  // make every row look the same.
  markdown: BookTextIcon,
  text: FileTextIcon,
  embed: FileTextIcon,
  download: FileIcon,
};

export const FileKindIcon: FC<{ path: string; className?: string }> = ({
  path,
  className,
}) => {
  const Icon = KIND_ICONS[guessIconKind(path)];
  return <Icon className={className} aria-hidden />;
};

/**
 * How far outside the scroller a tile starts loading. One panel-height of
 * warning, so scrolling meets pictures that are already there.
 */
const AHEAD = "300px";

/**
 * The picture for one tile, once it is worth having.
 *
 * **Nothing is fetched for a tile nobody can see.** This is half the fix for
 * the drawer: a conversation with three hundred photos in it used to read all
 * three hundred the moment the panel mounted. An `IntersectionObserver` turns
 * that into "the eight on screen", and it also means the *closed* panel costs
 * nothing — at `xl` it animates to `w-0`, and a tile inside a zero-width box
 * intersects nothing.
 *
 * The other half is `@/lib/thumbnails`, which makes sure the read happens once
 * rather than once per opening. See the note at the top of that file.
 */
function usePicture(
  path: string | undefined,
  wanted: boolean,
  version?: string | number,
): {
  box: (element: HTMLElement | null) => void;
  src?: string;
  onBroken: () => void;
} {
  // Where the tile is, so it can be watched. A callback ref rather than
  // `useRef`, because the effect below has to re-run when the node arrives.
  const [box, setBox] = useState<HTMLElement | null>(null);
  const [src, setSrc] = useState<string>();
  const [broken, setBroken] = useState(false);

  // The picture on screen must not outlive the file it is of. That means the
  // *version* as well as the path: the agent editing a photo in place leaves
  // the path alone, and holding the old square until the row happens to
  // unmount is the stale-thumbnail bug this all exists to avoid.
  const identity = path === undefined ? undefined : `${path} ${version ?? ""}`;
  const shown = useRef<string | undefined>(identity);
  if (shown.current !== identity) {
    shown.current = identity;
    if (src !== undefined) setSrc(undefined);
    if (broken) setBroken(false);
  }

  const live = path !== undefined && wanted && !broken;

  useEffect(() => {
    if (!live || !box || src !== undefined) return;
    if (!thumbnailsWork()) {
      // No decode pipeline here — jsdom in the tests, an old browser in the
      // wild. The original file is what this component always used to draw,
      // and it still works; it is only expensive.
      setSrc(fileUrl(path));
      return;
    }

    let alive = true;
    const start = () => {
      thumbnail(path, version).then(
        (made) => alive && setSrc(made),
        () => alive && setBroken(true),
      );
    };

    if (typeof IntersectionObserver !== "function") {
      start();
      return () => {
        alive = false;
      };
    }

    const watcher = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        watcher.disconnect();
        start();
      },
      { rootMargin: AHEAD },
    );
    watcher.observe(box);
    return () => {
      alive = false;
      watcher.disconnect();
    };
  }, [live, box, path, version, src]);

  return {
    box: setBox,
    src: broken ? undefined : src,
    onBroken: () => setBroken(true),
  };
}

/**
 * A file as a small square: its picture if it has one, its icon otherwise.
 *
 * **The icon is always drawn, and the picture sits on top of it.** That is what
 * makes a failure free: an image that will not load reveals what was already
 * painted underneath rather than swapping one element for another, so there is
 * no flicker and no second layout. It is also what makes the picture arriving
 * late free, since the tile is never empty while it is being made.
 *
 * **A failed picture is remembered, never removed from the DOM.** The obvious
 * version — `event.currentTarget.remove()` — takes a node out of the document
 * that React still has in its tree, and the bill arrives later: unmounting the
 * row throws `NotFoundError` from `removeChild` and the error boundary replaces
 * the entire app with a stack trace. A path here is a record of what happened
 * rather than a promise the file is still there, so a missing file is the
 * *common* case and a white screen was one stale row away. The remembering now
 * happens a level down, in the thumbnail cache, so it also survives the row
 * being unmounted and drawn again.
 *
 * Both surfaces that draw one — the files drawer's list and a user message's
 * attachments — had their own copy of all of the above.
 */
export const FileThumbnail: FC<{
  /** What to classify by. A file name is enough; a whole path is fine too. */
  name: string;
  /** Where to fetch the picture from. **Omit for a file that is gone**, or one
   *  whose durable path has not been read back yet — the icon still draws. */
  path?: string;
  /** The caller already knows this is an image, whatever the name suggests —
   *  a stored attachment carries its modality, which beats guessing at an
   *  extension. Widens the guess rather than replacing it. */
  image?: boolean;
  /** When this file was last touched, if the caller knows. It is part of the
   *  cache key, and it is the whole reason a rewritten file cannot show the
   *  thumbnail of what used to be at that path. */
  version?: string | number;
  /** The box: size, radius, and any shadow. */
  className?: string;
  /** The icon inside it, which does not scale with the box. */
  iconClassName?: string;
}> = ({ name, path, image = false, version, className, iconClassName }) => {
  const wanted = image || guessKind(name) === "image";
  const { box, src, onBroken } = usePicture(path, wanted, version);

  return (
    <span
      ref={box}
      className={cn(
        "bg-muted/60 relative flex shrink-0 items-center justify-center overflow-hidden border",
        className,
      )}
    >
      <FileKindIcon
        path={name}
        className={cn("text-muted-foreground", iconClassName)}
      />
      {src && (
        <img
          src={src}
          alt=""
          loading="lazy"
          decoding="async"
          // Only reachable on the fallback path, where `src` is the original
          // file: a thumbnail that was made is a picture that decoded.
          onError={onBroken}
          className="absolute inset-0 size-full object-cover"
        />
      )}
    </span>
  );
};
