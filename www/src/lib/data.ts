// Tiny data layer. Every JSON the React side reads lives under
// `out/` (served at `/data/` in dev by vite.config.ts and copied into
// `dist/data/` for production).
//
// Loads are memoised so re-renders don't refetch.

import type { IndexDoc, IndexCell } from "./types";

const cache = new Map<string, Promise<unknown>>();

export function dataUrl(rel: string): string {
  // Strip any leading slash and prefix with /data/.
  return `/data/${rel.replace(/^\/+/, "")}`;
}

export async function loadJson<T>(rel: string): Promise<T> {
  const url = dataUrl(rel);
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`fetch ${url} -> ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(url) as Promise<T>;
}

export function loadIndex(): Promise<IndexDoc> {
  return loadJson<IndexDoc>("_index.json");
}

// Find one cell by (model, script, n_max, periods). Most of the site
// asks for the paper-default cell (n_max=100, periods=[2,5,10,100]).
export function findCell(
  cells: IndexCell[],
  match: { model?: string; script?: string; n_max?: number; periods?: number[] },
): IndexCell | undefined {
  return cells.find((c) =>
    (match.model    === undefined || c.model   === match.model) &&
    (match.script   === undefined || c.script  === match.script) &&
    (match.n_max    === undefined || c.n_max   === match.n_max) &&
    (match.periods  === undefined ||
      (c.periods.length === match.periods.length &&
       c.periods.every((p, i) => p === match.periods![i]))),
  );
}

// Stable model and script orderings, used by selectors and the rho matrix.
// Order groups by family: EleutherAI, Meta, Google, AI2, Alibaba.
export const MODEL_ORDER = [
  "EleutherAI/pythia-6.9b",
  "EleutherAI/gpt-j-6b",
  "meta-llama/Llama-3.1-8B",
  "google/gemma-4-E4B",
  "google/gemma-4-31B",
  "allenai/Olmo-3-1125-32B",
  "Qwen/Qwen2.5-7B",
  "Qwen/Qwen2.5-32B",
];

export const MODEL_LABEL: Record<string, string> = {
  "EleutherAI/pythia-6.9b":    "Pythia-6.9B",
  "EleutherAI/gpt-j-6b":       "GPT-J-6B",
  "meta-llama/Llama-3.1-8B":   "Llama-3.1-8B",
  "google/gemma-4-E4B":        "Gemma-4-E4B",
  "google/gemma-4-31B":        "Gemma-4-31B",
  "allenai/Olmo-3-1125-32B":   "OLMo-3-32B",
  "Qwen/Qwen2.5-7B":           "Qwen2.5-7B",
  "Qwen/Qwen2.5-32B":          "Qwen2.5-32B",
};

// Shared model colour palette used by LayerSweep, RhoCkaScatter, etc.
// Order matches MODEL_ORDER so an index-based lookup is also valid.
export const MODEL_COLOR: Record<string, string> = {
  "EleutherAI/pythia-6.9b":    "#0369a1", // sky-700
  "EleutherAI/gpt-j-6b":       "#4338ca", // indigo-700
  "meta-llama/Llama-3.1-8B":   "#c2410c", // orange-700
  "google/gemma-4-E4B":        "#15803d", // green-700
  "google/gemma-4-31B":        "#7c3aed", // violet-600
  "allenai/Olmo-3-1125-32B":   "#be185d", // pink-700
  "Qwen/Qwen2.5-7B":           "#0891b2", // cyan-600
  "Qwen/Qwen2.5-32B":          "#a16207", // amber-700
};

export const SCRIPT_ORDER = [
  "latin", "arabic", "persian", "devanagari", "thai",
  "chinese", "binary", "hexadecimal",
  "greek", "hebrew", "roman", "babylonian",
] as const;

export const SCRIPT_LABEL: Record<string, string> = {
  latin:        "Latin",
  arabic:       "Arabic-Indic",
  persian:      "Persian",
  devanagari:   "Devanagari",
  thai:         "Thai",
  chinese:      "CJK digit string",
  binary:       "Binary",
  hexadecimal:  "Hexadecimal",
  greek:        "Greek alphabetic",
  hebrew:       "Hebrew alphabetic",
  roman:        "Roman",
  babylonian:   "Babylonian cuneiform",
};

export const SCRIPT_EXAMPLE: Record<string, string> = {
  latin:        "23",
  arabic:       "٢٣",
  persian:      "۲۳",
  devanagari:   "२३",
  thai:         "๒๓",
  chinese:      "二三",
  binary:       "10111",
  hexadecimal:  "17",
  greek:        "κγ",
  hebrew:       "כג",
  roman:        "XXIII",
  babylonian:   "𒌋𒌋𒁹𒁹𒁹",
};
