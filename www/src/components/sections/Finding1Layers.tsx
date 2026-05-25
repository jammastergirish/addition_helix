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
        First test of the diagnostic: does it confirm that on the paper's
        own cells, the helix is genuinely there? Yes — the original
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
    </section>
  );
}
