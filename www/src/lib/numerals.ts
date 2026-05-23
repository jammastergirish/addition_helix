// Render an integer in each of the 8 numeral systems we test.
// Ports the rendering logic in main.py (`format_number` and friends) so
// the React side can render any number live without round-tripping
// through Python. Cap is 599 (= one Babylonian sexagesimal wraparound +
// nine more, which is the largest number any chart in the site uses).

export type ScriptKey =
  | "latin" | "arabic" | "persian" | "devanagari"
  | "chinese" | "greek" | "roman" | "babylonian";

// ─── positional base-10 ─────────────────────────────────────────────

export const toLatin = (n: number) => String(n);

const shiftedDigits = (n: number, base: number) =>
  String(n).split("").map((d) => String.fromCharCode(base + +d)).join("");

export const toArabic     = (n: number) => shiftedDigits(n, 0x0660);
export const toPersian    = (n: number) => shiftedDigits(n, 0x06F0);
export const toDevanagari = (n: number) => shiftedDigits(n, 0x0966);

const CHINESE = "〇一二三四五六七八九";
export const toChinese = (n: number) =>
  String(n).split("").map((d) => CHINESE[+d]).join("");

// ─── additive ───────────────────────────────────────────────────────

const ROMAN_PAIRS: [number, string][] = [
  [1000, "M"], [900, "CM"], [500, "D"], [400, "CD"],
  [100, "C"],  [90, "XC"],  [50, "L"],  [40, "XL"],
  [10, "X"],   [9, "IX"],   [5, "V"],   [4, "IV"], [1, "I"],
];
export function toRoman(n: number): string {
  if (n === 0) return "nulla";
  let out = "";
  for (const [v, s] of ROMAN_PAIRS) {
    while (n >= v) { out += s; n -= v; }
  }
  return out;
}

const GREEK_UNITS    = ["", "α", "β", "γ", "δ", "ε", "ϛ", "ζ", "η", "θ"];
const GREEK_TENS     = ["", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ϟ"];
const GREEK_HUNDREDS = ["", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω", "ϡ"];
export function toGreek(n: number): string {
  if (n === 0) return "Ø";
  if (n < 0 || n > 999) return "?";
  const h = Math.floor(n / 100);
  const t = Math.floor((n % 100) / 10);
  const u = n % 10;
  return GREEK_HUNDREDS[h] + GREEK_TENS[t] + GREEK_UNITS[u];
}

// ─── mixed: positional base-60, additive within each column ─────────

const BAB_ONE  = "\u{12079}"; // 𒁹
const BAB_TEN  = "\u{1230B}"; // 𒌋
const BAB_ZERO = "\u{1244A}"; // 𒑊

function babColumn(v: number): string {
  if (v === 0) return BAB_ZERO;
  const tens = Math.floor(v / 10);
  const ones = v % 10;
  return BAB_TEN.repeat(tens) + BAB_ONE.repeat(ones);
}

export function toBabylonian(n: number): string {
  if (n === 0) return BAB_ZERO;
  if (n < 0) return "?";
  const columns: number[] = [];
  let rest = n;
  while (rest > 0) {
    columns.push(rest % 60);
    rest = Math.floor(rest / 60);
  }
  columns.reverse();
  return columns.map(babColumn).join(" ");
}

// ─── glue ───────────────────────────────────────────────────────────

export const RENDER: Record<ScriptKey, (n: number) => string> = {
  latin:      toLatin,
  arabic:     toArabic,
  persian:    toPersian,
  devanagari: toDevanagari,
  chinese:    toChinese,
  greek:      toGreek,
  roman:      toRoman,
  babylonian: toBabylonian,
};
