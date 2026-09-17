import { expect, it } from "vitest";
import { insertFormPath } from "@/lib/form-paths";

it("replaces only the selected text with the literal host path", () => {
  expect(insertFormPath("Read THIS please", "C:\\My files\\note.txt", "text", 5, 9))
    .toBe("Read C:\\My files\\note.txt please");
});

it("appends to JSON path lists and escapes Windows paths", () => {
  const path = 'C:\\My files\\a"b.txt';
  expect(JSON.parse(insertFormPath('["/existing"]', path, "json", 0, 0)))
    .toEqual(["/existing", path]);
  expect(JSON.parse(insertFormPath("", path, "json", 0, 0))).toEqual([path]);
});

it("inserts a JSON string at the selection in an unfinished object", () => {
  expect(insertFormPath('{"path": }', "C:\\work", "json", 9, 9))
    .toBe('{"path": "C:\\\\work"}');
});
