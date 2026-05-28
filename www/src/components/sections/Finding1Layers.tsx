import { useEffect, useState } from "react";
import type { IndexDoc, LayerSweepDoc } from "../../lib/types";
import { findCell, loadJson, MODEL_ORDER, MODEL_LABEL, MODEL_COLOR } from "../../lib/data";
import { LayerSweep } from "../charts/LayerSweep";
import { fmtPct } from "../../lib/svg";

interface Props { index: IndexDoc | null; }

const LATIN_DEFAULT = { script: "latin", n_max: 100, periods: [2, 5, 10, 100] };
const METRICS = ["pc1_r2", "helix_r2", "helix_over_pca"] as const;
type Metric = typeof METRICS[number];
const METRIC_LABEL: Record<Metric, string> = {
  pc1_r2:         "PC1 R²",
  helix_r2:       "helix R²",
  helix_over_pca: "helix / K-d PCA",
};

export function Finding1Layers({ index }: Props) {
  const [docs, setDocs] = useState<{ label: string; doc: LayerSweepDoc; color?: string }[]>([]);
  const [metric, setMetric] = useState<Metric>("helix_r2");

  useEffect(() => {
    if (!index) return;
    const cells = MODEL_ORDER.map((m) =>
      findCell(index.cells, { model: m, ...LATIN_DEFAULT }),
    );
    Promise.all(
      cells.map(async (c) => c?.paths.layer_sweep
        ? { label: MODEL_LABEL[c.model], color: MODEL_COLOR[c.model],
            doc: await loadJson<LayerSweepDoc>(c.paths.layer_sweep) }
        : null,
      ),
    ).then((arr) => setDocs(arr.filter(Boolean) as any));
  }, [index]);

  return (
    <section className="prose-body">
      <h2 className="section-heading">
        Demo: eight architectures, very different depths
      </h2>

      <p>
        First sanity check: do the measurements confirm that on the
        paper's own cells, the helix is genuinely there? Yes — the original
        finding survives a wider model panel. The paper studies
        Pythia-6.9B, GPT-J-6B, and Llama-3.1-8B; I keep all three and add
        Gemma 4 (E4B and 31B), OLMo-3-32B, and Qwen 2.5 (7B and 32B), so
        the matrix is a true superset of the paper's model set. All eight
        converge on the same Latin-digit geometry:
      </p>

      <table className="article-table mt-6">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">helix R²</th>
            <th className="text-right">helix / PCA</th>
            <th className="text-right">peak layer</th>
            <th className="text-right">total layers</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_ORDER.map((model) => {
            const c = index ? findCell(index.cells, { model, ...LATIN_DEFAULT }) : undefined;
            return (
              <tr key={model}>
                <td className="font-medium">{MODEL_LABEL[model]}</td>
                <td className="numeric text-right">{fmtPct(c?.helix_r2_peak)}</td>
                <td className="numeric text-right">{fmtPct(c?.helix_over_pca_peak)}</td>
                <td className="numeric text-right">{c?.peak_layer ?? "—"}</td>
                <td className="numeric text-right">{c?.n_layers ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p className="mt-6">
        But <em>where</em> the helix lives in the stack varies dramatically
        across models. <strong>Pythia</strong> rises monotonically and peaks
        at the literal final layer (L32 of 32). <strong>GPT-J</strong> —
        the paper's third model, restored here — does the same: peak at
        L28 of 28, ρ = 0.41, helix R² = 0.49. It sits cleanly between
        Pythia (ρ = 0.34) and Llama (ρ = 0.39), confirming the original
        paper's "depth substantially builds the helix" finding within its
        own model family.{" "}
        <strong>Llama</strong> peaks mid-stack at L15 of 32.{" "}
        <strong>Gemma-4-E4B</strong> shows two regimes: strong early
        structure at L0-L5, a dip around L6-L7, then a broad peak at L17
        of 42. <strong>Gemma-4-31B</strong> sits around 0.5 across most of
        its 60-layer stack with a sharp lift to 0.70 at L29 before falling
        back. <strong>OLMo-3-32B</strong> rises gradually through its
        mid-stack to a peak at L23 of 64. <strong>Qwen 2.5</strong> is the
        most extreme: both sizes peak in the first few layers — Qwen-7B
        at <strong>L0</strong> (the literal embedding output, before any
        block has run), Qwen-32B at L3 of 64 — and gently decay through
        the rest of the stack. On Pythia and Llama, depth has to{" "}
        <em>build</em> the helix from a low L0; on Qwen, the helix is
        essentially complete at the embedding lookup. Across eight models
        the only invariant is that there is no invariant.{" "}
        <strong>"Read the middle layer" is a Pythia-ism; for any new model,
        a layer sweep is mandatory.</strong> The layer sweep is also what
        makes the ρ diagnostic possible — without identifying each model's
        peak layer, there's no peak to ratio L=0 against.
      </p>

      <div className="my-6 rounded-lg border border-ink/10 bg-white p-4">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          {METRICS.map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={[
                "rounded-full border px-3 py-1 font-medium tracking-wide transition",
                metric === m
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-ink/15 text-ink-mute hover:border-ink/30 hover:text-ink",
              ].join(" ")}
            >
              {METRIC_LABEL[m]}
            </button>
          ))}
        </div>
        {docs.length === 0 ? (
          <div className="flex h-48 items-center justify-center text-sm text-ink-mute">loading layer sweeps…</div>
        ) : (
          <LayerSweep docs={docs} metric={metric} />
        )}
      </div>

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Deep dive: <em>reading a layer sweep</em>
        </summary>

        <p>
          For one cell, the layer sweep gives us three curves indexed
          by layer L (from 0 to n_layers). Each tells you something
          different about what's happening at that depth.
        </p>

        <p className="mt-3">
          <strong>PC1 R² ("the spine").</strong> PCA finds the direction
          of maximum variance across the 100 integer-vectors at this
          layer. Project all 100 vectors onto that single direction → 100
          scalar values. Fit a line <span className="font-mono">pc1(a) = m·a + b</span>{" "}
          and report R².
        </p>
        <ul className="mt-1 list-disc pl-6 space-y-1">
          <li>
            High PC1 R² (~1) = the most-variance direction <em>is</em>{" "}
            a number line. The model's most prominent way of
            distinguishing integers at this layer is by magnitude.
          </li>
          <li>
            Low PC1 R² = the dominant variance direction is something
            else — some grammatical feature, a tokenizer artifact,
            anything that varies a lot across the 100 inputs but isn't
            their integer value.
          </li>
        </ul>

        <p className="mt-3">
          For Pythia/Latin at L=0, PC1 R² is moderate: embeddings for
          0–99 have <em>some</em> linear structure but not a clean
          ramp. By the peak layer, PC1 R² is ~0.95: depth has
          organised the magnitude axis into a clean number line.
        </p>

        <p className="mt-3">
          <strong>Helix R².</strong> The 9-feature trig basis B(a)
          fitted to the 4096-d residual stream at this layer.
          Variance-weighted R² across hidden dimensions. This is the
          number K&amp;T's analysis reports at the peak layer.
        </p>

        <p className="mt-3">
          <strong>Quality ratio (helix / K-d PCA).</strong> Helix R²
          alone is just a score; it doesn't tell you whether the helix
          is <em>the</em> dominant 9-d structure or one of many 9-d
          fits that could work. So we compute the ceiling: what is the
          best possible 9-d fit to H at this layer? That's the
          Eckart–Young theorem — take the top-9 principal components
          of H, reconstruct, report R². No 9-feature basis can fit
          better.
        </p>

        <p className="mt-3">
          The ratio = helix R² / K-d PCA R² tells you how dominant
          the helix subspace is. Ratio of 0.85 means the trig basis
          captures 85% of what any 9-d basis could capture — so it
          isn't just <em>a</em> good 9-d fit, it basically{" "}
          <em>is</em> the 9-d structure.
        </p>

        <p className="mt-3">
          <strong>Where the helix lives in the stack — why we sweep.</strong>{" "}
          K&amp;T originally read at the middle layer
          (<span className="font-mono">n_layers // 2</span>). For
          Pythia (32 layers), that's L=16. For Llama (32 layers), L=16.
          Both happen to have peaks around there. "Read the middle"
          was a fine heuristic for K&amp;T's three models. It is the{" "}
          <em>wrong</em> heuristic for any of the five we added:
        </p>

        <ul className="mt-3 list-disc pl-6 space-y-1">
          <li>Pythia: peaks at L=32 of 32 (literal final layer).</li>
          <li>GPT-J: peaks at L=28 of 28.</li>
          <li>Llama: peaks at L=15 of 32 (mid-stack).</li>
          <li>
            Qwen2.5-7B/Latin: peaks at <strong>L=0</strong> — the helix
            is complete before any block has run. Middle-layer reading
            would miss the peak entirely.
          </li>
          <li>Qwen2.5-32B/Latin: peaks at L=3 of 64.</li>
          <li>
            Gemma-4-31B: moderate R² across the stack with a sharp
            spike to peak at L=29 of 60.
          </li>
          <li>Gemma-4-E4B: double-peaked, global max at L=17 of 42.</li>
          <li>OLMo-3-32B: rises gradually to a peak at L=23 of 64.</li>
        </ul>

        <p className="mt-3">
          Without the sweep we couldn't even compute ρ — there'd be
          no <span className="font-mono">max_L R²</span> to ratio
          L=0 against. The sweep also lets us auto-target the
          standard figures (FFT+PC1, circles, 3D helix, 2D PCA) at
          each model's actual peak instead of K&amp;T's middle-layer
          default. So the 3D helix you rotated above is read at
          whichever layer the model actually puts its cleanest helix,
          rather than the layer K&amp;T happened to read on Pythia.
        </p>

        <p className="mt-3">
          <strong>How to read a sweep plot.</strong> Toggle the metric
          buttons above. From the shape you can tell:
        </p>

        <ul className="mt-3 list-disc pl-6 space-y-1">
          <li>
            <em>Pythia/Latin</em>: helix R² rises monotonically from
            low at L=0 to high at the final layer. The textbook "depth
            builds the helix" pattern.
          </li>
          <li>
            <em>Qwen2.5-7B/Latin</em>: helix R² starts high at L=0,
            peaks immediately, decays gently. The helix exists in
            the embedding+pooling output and depth doesn't add to it.
          </li>
          <li>
            <em>Gemma-4-31B/Latin</em>: helix R² sits around 0.5 for
            most layers with a pronounced spike to 0.7 at L=29 then
            falls back. Depth is <em>doing something</em> at L=29
            specifically.
          </li>
        </ul>

        <p className="mt-3">
          The sweep also reveals what ρ alone would hide. A cell with
          ρ = 0.6 could have a smooth rise from L=0 to peak (depth
          gradually refines) <em>or</em> a flat-then-spike pattern
          (depth suddenly builds something at one layer). The shape
          of the curve carries information ρ collapses.
        </p>
      </details>
    </section>
  );
}
