import { useEffect, useState } from "react";
import type { IndexDoc, LayerSweepDoc } from "../../lib/types";
import { findCell, loadJson, MODEL_ORDER, MODEL_LABEL } from "../../lib/data";
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
    const colors = ["#0369a1", "#c2410c", "#15803d", "#7c3aed", "#be185d"];
    Promise.all(
      cells.map(async (c, i) => c?.paths.layer_sweep
        ? { label: MODEL_LABEL[c.model], color: colors[i],
            doc: await loadJson<LayerSweepDoc>(c.paths.layer_sweep) }
        : null,
      ),
    ).then((arr) => setDocs(arr.filter(Boolean) as any));
  }, [index]);

  return (
    <section className="prose-body">
      <h2 className="section-heading">
        It replicates in five architectures, at very different depths
      </h2>

      <p>
        The paper studies Pythia-6.9B, GPT-J-6B, and Llama-3.1-8B — three
        decoder-only transformers in the same broad lineage, all read at the
        middle layer on Latin/0–99. Substituting Gemma 4 for GPT-J and adding
        OLMo-3-32B, all five converge on the same Latin-digit geometry:
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
        — five models, five distinct profiles. <strong>Pythia</strong> rises
        monotonically and peaks at the literal final layer (L32 of 32).{" "}
        <strong>Llama</strong> peaks mid-stack at L15 of 32.{" "}
        <strong>Gemma-4-E4B</strong> shows two regimes: strong early
        structure at L0-L5, a dip around L6-L7, then a broad peak at L17
        of 42. <strong>Gemma-4-31B</strong> sits around 0.5 across most of
        its 60-layer stack with a sharp lift to 0.70 at L29 before falling
        back. <strong>OLMo-3-32B</strong> rises gradually through its
        mid-stack to a peak at L23 of 64.{" "}
        <strong>"Read the middle layer" is a Pythia-ism; for any new model,
        a layer sweep is mandatory.</strong>
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
