import { describe, expect, it } from "vitest";

import { delimiterFor, parseDelimited } from "@/lib/csv";

const csv = (text: string) => parseDelimited(text, ",");

describe("parseDelimited", () => {
  it("reads plain rows", () => {
    expect(csv("a,b,c\n1,2,3")).toEqual([
      ["a", "b", "c"],
      ["1", "2", "3"],
    ]);
  });

  it("keeps empty cells rather than collapsing them", () => {
    expect(csv("a,,c")).toEqual([["a", "", "c"]]);
  });

  it("does not invent a final row from a trailing newline", () => {
    expect(csv("a,b\n1,2\n")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("takes the delimiter inside a quoted cell literally", () => {
    expect(csv('name,note\n"Doe, Jane",hi')).toEqual([
      ["name", "note"],
      ["Doe, Jane", "hi"],
    ]);
  });

  it("reads a doubled quote as one quote", () => {
    expect(csv('a,"she said ""no""",c')).toEqual([
      ["a", 'she said "no"', "c"],
    ]);
  });

  it("keeps a newline inside a quoted cell in the same row", () => {
    expect(csv('a,"line one\nline two",c')).toEqual([
      ["a", "line one\nline two", "c"],
    ]);
  });

  it("treats CRLF as one break", () => {
    expect(csv("a,b\r\n1,2")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("treats a lone CR as a break", () => {
    expect(csv("a,b\r1,2")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("ends an unterminated quote at the end of the file", () => {
    // Wonky, but readable — which beats refusing the whole file.
    expect(csv('a,"never closed')).toEqual([["a", "never closed"]]);
  });

  it("reads nothing out of nothing", () => {
    expect(csv("")).toEqual([]);
  });

  it("splits on tabs when asked", () => {
    expect(parseDelimited("a\tb\n1\t2", "\t")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });
});

describe("delimiterFor", () => {
  it("picks a tab for .tsv, with or without the dot", () => {
    expect(delimiterFor(".tsv")).toBe("\t");
    expect(delimiterFor("TSV")).toBe("\t");
  });

  it("falls back to a comma for everything else", () => {
    expect(delimiterFor(".csv")).toBe(",");
    expect(delimiterFor("")).toBe(",");
  });
});
