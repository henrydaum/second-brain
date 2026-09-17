/**
 * Delimited text, as rows of cells.
 *
 * **A split on commas is not a CSV parser, and the difference shows up on real
 * files rather than on made-up ones.** A quoted cell may contain the delimiter,
 * a doubled quote, or a line break — all three are ordinary in anything a
 * spreadsheet exported — and each one makes the naive version silently produce
 * a table with the wrong shape. Wrong is worse than absent here: a misparsed
 * table looks entirely plausible.
 *
 * Written by hand rather than pulled in, because the whole of the specification
 * this needs is the four rules below, and a dependency for them would be a
 * dependency to keep.
 *
 * The parsing happens in the browser on purpose. `.csv` answers modality
 * `"text"` and always will — the bundled `parse_text` claims the extension and
 * the first registration wins — so there is no Request that hands back a parsed
 * sheet. That is the right answer to the question modality asks (*how should
 * the model ingest this*) and the wrong one for the question a viewer asks
 * (*how should a person look at this*), which is why this exists.
 */

/** The separator a file's extension implies. Unknown extensions get a comma,
 *  which is the only guess worth making. */
export function delimiterFor(extension: string): string {
  return extension.replace(/^\./, "").toLowerCase() === "tsv" ? "\t" : ",";
}

/**
 * Delimited text → rows of cells.
 *
 * The four rules, which are RFC 4180's minus the parts nothing emits:
 *
 * - A cell wrapped in `"` may contain the delimiter and newlines verbatim.
 * - Inside such a cell, `""` is a literal quote.
 * - Line endings are `\n` or `\r\n`, and a lone `\r` is treated as one too —
 *   old Mac exports still exist and cost nothing to accept.
 * - A trailing newline ends the file rather than starting an empty final row.
 *
 * Unterminated quotes are not an error: the rest of the file becomes the last
 * cell. A viewer showing a wonky final row beats one refusing to show a file it
 * could mostly read.
 */
export function parseDelimited(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  // Whether anything at all has been seen since the last row break — the test
  // that tells a trailing newline from a genuine empty row.
  let started = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    if (quoted) {
      if (char === '"') {
        // A doubled quote is one quote; a single one closes the cell.
        if (text[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        cell += char;
      }
      continue;
    }

    if (char === '"' && cell === "") {
      quoted = true;
      started = true;
      continue;
    }

    if (char === delimiter) {
      row.push(cell);
      cell = "";
      started = true;
      continue;
    }

    if (char === "\n" || char === "\r") {
      // `\r\n` is one break, not two.
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      started = false;
      continue;
    }

    cell += char;
    started = true;
  }

  // Whatever is still in hand is the last row — unless the file simply ended
  // with a newline, in which case there is nothing left to add.
  if (started || cell !== "" || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }

  return rows;
}
