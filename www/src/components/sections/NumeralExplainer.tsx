import { useMemo, useState } from "react";
import { RENDER, type ScriptKey } from "../../lib/numerals";

interface SystemDef {
  key: ScriptKey;
  name: string;
  /** Structure-class label, surfaced on the card chip. */
  structure: string;
  /** One-sentence description shown under the rendered numeral. */
  blurb: React.ReactNode;
  /** Decomposition for the chosen integer — recomputed live. */
  decompose?: (n: number) => string;
}

const POSITIONAL_BASE_10: SystemDef[] = [
  {
    key: "latin", name: "Latin", structure: "positional · base 10",
    blurb: "Place-value notation. Each position is a power of 10.",
    decompose: (n) => decimalDecomp(n),
  },
  {
    key: "arabic", name: "Arabic-Indic", structure: "positional · base 10",
    blurb: "The original Arabic glyphs, same structure as Latin. Numbers are written most-significant-first despite Arabic script flowing right-to-left.",
    decompose: (n) => decimalDecomp(n),
  },
  {
    key: "persian", name: "Persian", structure: "positional · base 10",
    blurb: "Visually similar to Arabic-Indic but a different Unicode block — some digits (٤/۴) differ subtly.",
    decompose: (n) => decimalDecomp(n),
  },
  {
    key: "devanagari", name: "Devanagari", structure: "positional · base 10",
    blurb: "Used across Indian languages — Hindi, Sanskrit, Marathi.",
    decompose: (n) => decimalDecomp(n),
  },
  {
    key: "chinese", name: "CJK digit string", structure: "positional · base 10",
    blurb: <>A digit-by-digit, place-value rendering using CJK numeral glyphs — <em>not</em> ordinary written Chinese number syntax (which would be <span className="font-mono">二十三</span>, "two-tens-three"). I use the positional form so this script stays comparable to Latin / Arabic-Indic / Devanagari.</>,
    decompose: (n) => decimalDecomp(n),
  },
  {
    key: "thai", name: "Thai", structure: "positional · base 10",
    blurb: <>Thai numerals are direct base-10 positional digits (<span className="font-mono">๐</span>–<span className="font-mono">๙</span>), structurally identical to Latin / Arabic-Indic / Devanagari but a separate Unicode block.</>,
    decompose: (n) => decimalDecomp(n),
  },
];

const POSITIONAL_OTHER_BASES: SystemDef[] = [
  {
    key: "binary", name: "Binary", structure: "positional · base 2",
    blurb: <>Just two digit glyphs (0, 1). Natural Fourier periods are 2, 4, 8, 16, 32, 64 — the paper basis catches only T = 2.</>,
    decompose: (n) => baseDecomp(n, 2),
  },
  {
    key: "hexadecimal", name: "Hexadecimal", structure: "positional · base 16",
    blurb: <>Sixteen digit glyphs (0-9, a-f). Natural periods are 16 and 256; the paper basis (built for base 10) catches almost none of the structure.</>,
    decompose: (n) => baseDecomp(n, 16),
  },
];

const ADDITIVE: SystemDef[] = [
  {
    key: "greek", name: "Greek alphabetic", structure: "additive (Milesian)",
    blurb: <>Each letter has a fixed value: α=1, β=2, …, ι=10, κ=20, …, ρ=100, σ=200. Adjacent integers can have unrelated glyphs (9 = θ, 10 = ι).</>,
    decompose: (n) => greekDecomp(n),
  },
  {
    key: "hebrew", name: "Hebrew alphabetic", structure: "additive (gematria)",
    blurb: <>Like Greek. Units א=1…ט=9, tens י=10…צ=90 (so 23 = כג = 20+3). Special cases at 15 (טו, not יה) and 16 (טז, not יו) avoid spelling part of the divine name. Hebrew is RTL — the storage-largest letter is written first but appears on the right.</>,
    decompose: (n) => hebrewDecomp(n),
  },
  {
    key: "roman", name: "Roman", structure: "additive · subtractive shortcuts",
    blurb: <>Sum letter values. Subtractive shortcuts at 4 (IV), 9 (IX), 40 (XL), 90 (XC), etc.</>,
    decompose: (n) => romanDecomp(n),
  },
];

const MIXED: SystemDef[] = [
  {
    key: "babylonian", name: "Babylonian cuneiform", structure: "positional · base 60 · additive within column",
    blurb: <>Each sexagesimal column is rendered additively with <span className="font-mono">𒌋</span>=10 and <span className="font-mono">𒁹</span>=1 wedges. At 60 a new column opens to the left; at 3600 another; and so on.</>,
    decompose: (n) => babylonianDecomp(n),
  },
];

// ── decomposition helpers (purely cosmetic text under each numeral) ──

function decimalDecomp(n: number): string {
  if (n === 0) return "= 0";
  const digits = String(n).split("").map(Number);
  const parts: string[] = [];
  digits.forEach((d, i) => {
    const power = digits.length - 1 - i;
    if (d === 0) return;
    if (power === 0) parts.push(`${d}`);
    else parts.push(`${d}·10${superscript(power)}`);
  });
  return `= ${parts.join(" + ")}`;
}

function greekDecomp(n: number): string {
  if (n === 0) return "(no Greek zero)";
  if (n > 999) return "(out of range)";
  const h = Math.floor(n / 100);
  const t = Math.floor((n % 100) / 10);
  const u = n % 10;
  const parts: string[] = [];
  if (h) parts.push(`${h * 100}`);
  if (t) parts.push(`${t * 10}`);
  if (u) parts.push(`${u}`);
  return `= ${parts.join(" + ")}`;
}

function hebrewDecomp(n: number): string {
  if (n === 0) return "= 0  (אפס, modern Hebrew word for zero)";
  if (n > 99) return "(out of range)";
  if (n === 15 || n === 16) {
    const inner = n === 15 ? "9 + 6" : "9 + 7";
    return `= ${inner}  (special: avoids ${n === 15 ? "יה" : "יו"})`;
  }
  const t = Math.floor(n / 10), u = n % 10;
  const parts: string[] = [];
  if (t) parts.push(`${t * 10}`);
  if (u) parts.push(`${u}`);
  return `= ${parts.join(" + ")}`;
}

function baseDecomp(n: number, base: number): string {
  if (n === 0) return "= 0";
  const digits: number[] = [];
  let rest = n;
  while (rest > 0) {
    digits.push(rest % base);
    rest = Math.floor(rest / base);
  }
  digits.reverse();
  const parts: string[] = [];
  digits.forEach((d, i) => {
    const power = digits.length - 1 - i;
    if (d === 0) return;
    if (power === 0) parts.push(`${d}`);
    else parts.push(`${d}·${base}${superscript(power)}`);
  });
  return `= ${parts.join(" + ")}`;
}

function romanDecomp(n: number): string {
  if (n === 0) return "(no Roman zero — 'nulla')";
  const parts: string[] = [];
  let rest = n;
  for (const [v, s] of [[1000,"M"],[900,"CM"],[500,"D"],[400,"CD"],[100,"C"],[90,"XC"],[50,"L"],[40,"XL"],[10,"X"],[9,"IX"],[5,"V"],[4,"IV"],[1,"I"]] as [number,string][]) {
    while (rest >= v) {
      parts.push(`${s}=${v}`);
      rest -= v;
    }
  }
  // Compress repeated letters: "I=1, I=1, I=1" -> "III = 3×1"
  return `= ${n} (sum of glyph values)`;
}

function babylonianDecomp(n: number): string {
  if (n === 0) return "(late-period zero placeholder)";
  const columns: { value: number; power: number }[] = [];
  let rest = n;
  let p = 0;
  while (rest > 0) {
    columns.push({ value: rest % 60, power: p });
    rest = Math.floor(rest / 60);
    p++;
  }
  columns.reverse();
  const parts = columns
    .filter((c) => c.value !== 0)
    .map((c) => c.power === 0
      ? `${c.value}`
      : `${c.value}·60${superscript(c.power)}`);
  return `= ${parts.join(" + ")}`;
}

function superscript(p: number): string {
  return String(p).split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]).join("");
}

// ─── component ──────────────────────────────────────────────────────

const N_MAX = 599;

export function NumeralExplainer() {
  const [n, setN] = useState(23);

  const setSafe = (v: number) => {
    if (Number.isNaN(v)) return;
    setN(Math.max(0, Math.min(N_MAX, Math.floor(v))));
  };

  // Threshold annotations: highlight when the input is at an interesting place.
  const flag = useMemo(() => {
    if (n === 60)  return { color: "bg-amber-100 text-amber-900 border-amber-300", text: "↑ at 60, Babylonian opens a new sexagesimal column" };
    if (n === 10)  return { color: "bg-sky-100 text-sky-900 border-sky-300",    text: "↑ at 10, base-10 positional systems switch from a single digit to two" };
    if (n === 4)   return { color: "bg-rose-100 text-rose-900 border-rose-300", text: "↑ at 4, Roman switches from III to IV (subtractive)" };
    if (n === 9)   return { color: "bg-rose-100 text-rose-900 border-rose-300", text: "↑ at 9, Roman switches from VIII to IX, Greek from θ to ι" };
    if (n === 15)  return { color: "bg-rose-100 text-rose-900 border-rose-300", text: "↑ at 15, Hebrew uses טו (9+6) instead of יה (10+5) to avoid divine-name spelling" };
    if (n === 16)  return { color: "bg-emerald-100 text-emerald-900 border-emerald-300", text: "↑ at 16, hex flips 'f'→'10', binary adds a 5th bit (10000); Hebrew uses טז (9+7)" };
    if (n === 2)   return { color: "bg-emerald-100 text-emerald-900 border-emerald-300", text: "↑ at 2, binary becomes two digits (10)" };
    if (n === 8)   return { color: "bg-emerald-100 text-emerald-900 border-emerald-300", text: "↑ at 8, binary becomes 4 digits (1000)" };
    if (n === 100) return { color: "bg-sky-100 text-sky-900 border-sky-300",    text: "↑ at 100, base-10 systems add a third digit; Greek introduces hundreds letters (ρ); hex becomes 3 digits (64)" };
    if (n === 256) return { color: "bg-emerald-100 text-emerald-900 border-emerald-300", text: "↑ at 256 = 16², hex opens a third column" };
    if (n === 3600) return { color: "bg-amber-100 text-amber-900 border-amber-300", text: "↑ at 3600 = 60², Babylonian opens a third column" };
    return null;
  }, [n]);

  return (
    <section className="prose-body pt-12">
      <h2 className="section-heading">The twelve numeral systems</h2>
      <p>
        I test the helix across twelve systems in four families: six{" "}
        <strong>positional base-10</strong> systems (Latin, Arabic-Indic,
        Persian, Devanagari, Thai, CJK digit string), two{" "}
        <strong>positional non-decimal</strong> systems (binary, hexadecimal),
        three <strong>additive</strong> systems (Greek alphabetic, Hebrew
        alphabetic, Roman), and Babylonian cuneiform — a{" "}
        <strong>hybrid</strong> that is positional at base 60 but additive
        within each sexagesimal column.
      </p>

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-3">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Pick a number and see how each system renders it
        </summary>

      <div className="my-2 rounded-lg border border-ink/15 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs font-medium uppercase tracking-wider text-ink-mute">
            number
          </label>
          <input
            type="number" min={0} max={N_MAX} value={n}
            onChange={(e) => setSafe(+e.target.value)}
            className="w-24 rounded border border-ink/15 bg-paper-warm/40 px-2 py-1 text-right font-mono text-sm"
          />
          <input
            type="range" min={0} max={N_MAX} value={n}
            onChange={(e) => setSafe(+e.target.value)}
            className="flex-1 min-w-[160px] accent-accent"
          />
          <span className="font-mono text-xs text-ink-mute">0 – {N_MAX}</span>
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5 text-xs">
          {[0, 4, 9, 10, 15, 16, 23, 60, 99, 100, 256, 599].map((m) => (
            <button
              key={m}
              onClick={() => setSafe(m)}
              className={[
                "rounded-full border px-2.5 py-0.5 font-mono transition",
                n === m
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-ink/15 text-ink-mute hover:border-ink/30 hover:text-ink",
              ].join(" ")}
            >
              {m}
            </button>
          ))}
        </div>

        {flag && (
          <div className={`mt-3 rounded border px-3 py-1.5 text-xs ${flag.color}`}>
            {flag.text}
          </div>
        )}

        <Group title="Positional base-10" systems={POSITIONAL_BASE_10} n={n} />
        <Group title="Positional, other bases" systems={POSITIONAL_OTHER_BASES} n={n} />
        <Group title="Additive (non-positional)" systems={ADDITIVE} n={n} />
        <Group title="Mixed: positional base-60, additive within each column" systems={MIXED} n={n} />
      </div>
      </details>
    </section>
  );
}

function Group({ title, systems, n }: { title: string; systems: SystemDef[]; n: number }) {
  return (
    <div className="mt-6">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-mute">
        {title}
      </h3>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {systems.map((s) => (
          <SystemCard key={s.key} system={s} n={n} />
        ))}
      </div>
    </div>
  );
}

function SystemCard({ system, n }: { system: SystemDef; n: number }) {
  const rendered = RENDER[system.key](n);
  const decomp = system.decompose?.(n);

  return (
    <div className="rounded-md border border-ink/10 bg-paper-warm/30 p-3 transition hover:bg-paper-warm/60">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-sm font-semibold text-ink">{system.name}</div>
      </div>
      <div className="mt-0.5 text-[10.5px] font-medium uppercase tracking-wider text-ink-mute">
        {system.structure}
      </div>

      <div
        className="my-3 break-words text-center font-mono text-2xl leading-tight text-ink"
        style={{ fontFeatureSettings: "'tnum'" }}
      >
        {rendered}
      </div>

      {decomp && (
        <div className="text-center text-[11px] font-mono text-ink-mute">
          {decomp}
        </div>
      )}

      <div className="mt-2 text-[12px] leading-snug text-ink-soft">
        {system.blurb}
      </div>
    </div>
  );
}
