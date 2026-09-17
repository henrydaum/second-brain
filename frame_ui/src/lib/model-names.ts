/**
 * How model names are actually spelled.
 *
 * **A hand-maintained list, on purpose.** Model IDs arrive lowercased and
 * hyphenated (`minimax/minimax-m3`), and no amount of parsing recovers that
 * the vendor writes it "MiniMax" — capitalisation inside a word is a fact
 * about a brand, not a rule about English. There is no feed to subscribe to,
 * so this table is the source of truth: add a line when a model shows up
 * spelled wrong in the picker.
 *
 * **Only the exceptions are listed.** Anything missing is title-cased, which
 * is already right for Claude, Sonnet, Gemini, Grok, Mistral, Phi, Nemotron,
 * Kimi, Falcon, Granite, Nova, and most of the field — so a model released
 * after this file was last touched renders plainly rather than wrongly. The
 * entries below are the words where title case gets it *wrong*, plus the
 * couple of brands (`llama`, `qwen`) kept for the reader's benefit even
 * though title case happens to agree.
 *
 * Keys are lowercase and match one whitespace/hyphen/underscore-separated word
 * of a model ID. Values are the spelling to show.
 */
export const MODEL_WORDS: Record<string, string> = {
  // ── Acronyms, wherever they appear ────────────────────────────────────────
  ai: "AI",
  api: "API",
  asr: "ASR",
  gpt: "GPT",
  llm: "LLM",
  moe: "MoE",
  ocr: "OCR",
  oss: "OSS",
  sql: "SQL",
  tts: "TTS",
  ui: "UI",
  vl: "VL",
  vlm: "VLM",
  // Gemma and friends tag instruction-tuned weights `-it`. In a model ID this
  // is never the English word.
  it: "IT",

  // ── Quantisation and precision tags ───────────────────────────────────────
  awq: "AWQ",
  bf16: "BF16",
  exl2: "EXL2",
  fp8: "FP8",
  fp16: "FP16",
  gguf: "GGUF",
  gptq: "GPTQ",
  int4: "INT4",
  int8: "INT8",
  nf4: "NF4",

  // ── Size suffixes ─────────────────────────────────────────────────────────
  xl: "XL",
  xxl: "XXL",

  // ── OpenAI ────────────────────────────────────────────────────────────────
  chatgpt: "ChatGPT",
  openai: "OpenAI",
  // The reasoning line is lowercase in OpenAI's own writing.
  o1: "o1",
  o3: "o3",
  o4: "o4",

  // ── Anthropic, Google, xAI, Meta ──────────────────────────────────────────
  codegemma: "CodeGemma",
  codellama: "CodeLlama",
  llama: "Llama",
  llamaguard: "LlamaGuard",
  medlm: "MedLM",
  medpalm: "MedPaLM",
  paligemma: "PaliGemma",
  palm: "PaLM",
  recurrentgemma: "RecurrentGemma",
  shieldgemma: "ShieldGemma",
  xai: "xAI",

  // ── Chinese labs ──────────────────────────────────────────────────────────
  chatglm: "ChatGLM",
  codegeex: "CodeGeeX",
  deepseek: "DeepSeek",
  ernie: "ERNIE",
  exaone: "EXAONE",
  glm: "GLM",
  internlm: "InternLM",
  mimo: "MiMo",
  minimax: "MiniMax",
  qvq: "QvQ",
  qwen: "Qwen",
  qwq: "QwQ",
  // ByteDance's computer-use line, `ui-tars`.
  tars: "TARS",

  // ── Open-weight families and fine-tunes ───────────────────────────────────
  bakllava: "BakLLaVA",
  bloom: "BLOOM",
  bloomz: "BLOOMZ",
  codegen: "CodeGen",
  dbrx: "DBRX",
  flan: "FLAN",
  lfm: "LFM",
  llava: "LLaVA",
  mpt: "MPT",
  neox: "NeoX",
  nousresearch: "NousResearch",
  olmo: "OLMo",
  openchat: "OpenChat",
  openelm: "OpenELM",
  openhermes: "OpenHermes",
  rwkv: "RWKV",
  santacoder: "SantaCoder",
  smollm: "SmolLM",
  stablecode: "StableCode",
  stablelm: "StableLM",
  starcoder: "StarCoder",
  tinyllama: "TinyLlama",
  wizardcoder: "WizardCoder",
  wizardlm: "WizardLM",
  wizardmath: "WizardMath",
  xgen: "XGen",
  xglm: "XGLM",

  // ── Gateways and embedding models ─────────────────────────────────────────
  bge: "BGE",
  gte: "GTE",
  openrouter: "OpenRouter",
  pplx: "PPLX",
};

/** A brand fused to its version with no separator: `glm4.6`, `qwen3`. */
const FUSED = /^([a-z]+)([\d.].*)$/;
/**
 * A number with a unit: `120b` parameters, `a3b` active ones, `8x7b` experts,
 * `128k` of context, `1m`. The unit is the part that is shouted — `8X7B` would
 * be wrong, so the `x` between counts stays down.
 */
const COUNT = /^a?\d+(\.\d+)?(x\d+(\.\d+)?)?[bmk]$/;

/** One word of a model ID, spelled the way its vendor spells it. */
export function spellModelWord(word: string): string {
  const lower = word.toLowerCase();
  const known = MODEL_WORDS[lower];
  if (known) return known;

  const fused = FUSED.exec(lower);
  const brand = fused && MODEL_WORDS[fused[1]];
  if (fused && brand) return `${brand}${fused[2]}`;

  if (COUNT.test(lower)) {
    return lower.replace(/[a-z]/g, (letter) =>
      letter === "x" ? letter : letter.toUpperCase(),
    );
  }

  return lower.charAt(0).toUpperCase() + lower.slice(1);
}
