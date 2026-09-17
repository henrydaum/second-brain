import { describe, expect, it } from "vitest";

import { spellModelWord } from "@/lib/model-names";

describe("spellModelWord", () => {
  it.each([
    ["deepseek", "DeepSeek"],
    ["Minimax", "MiniMax"],
    ["MINIMAX", "MiniMax"],
    ["gpt", "GPT"],
    ["qwq", "QwQ"],
    ["llava", "LLaVA"],
    ["awq", "AWQ"],
    ["it", "IT"],
    ["o3", "o3"],
  ])("spells the known word %s", (input, expected) => {
    expect(spellModelWord(input)).toBe(expected);
  });

  it.each([
    ["glm4.6", "GLM4.6"],
    ["qwen3", "Qwen3"],
    ["internlm3", "InternLM3"],
  ])("keeps a brand fused to its version readable: %s", (input, expected) => {
    expect(spellModelWord(input)).toBe(expected);
  });

  it.each([
    ["120b", "120B"],
    ["1.5b", "1.5B"],
    ["8x7b", "8x7B"],
    ["a3b", "A3B"],
    ["128k", "128K"],
    ["1m", "1M"],
  ])("shouts only the unit in a count: %s", (input, expected) => {
    expect(spellModelWord(input)).toBe(expected);
  });

  it.each([
    ["sonnet", "Sonnet"],
    ["nemotron", "Nemotron"],
    ["16e", "16e"],
    ["m3", "M3"],
    ["4o", "4o"],
    ["v3.1", "V3.1"],
    ["", ""],
  ])("title-cases anything unlisted: %s", (input, expected) => {
    expect(spellModelWord(input)).toBe(expected);
  });
});
