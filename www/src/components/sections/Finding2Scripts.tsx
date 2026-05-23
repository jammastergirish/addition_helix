import { useMemo, useState } from "react";
import type { IndexDoc, CirclesDoc, Helix3DDoc, PCA2DDoc } from "../../lib/types";
import {
  findCell, MODEL_ORDER, MODEL_LABEL,
  SCRIPT_ORDER, SCRIPT_LABEL, SCRIPT_EXAMPLE,
} from "../../lib/data";
import { ChartFrame } from "../ChartFrame";
import { CirclePanel } from "../charts/CirclePanel";
import { HelixViewer3D } from "../charts/HelixViewer3D";
import { PCA2DScatter } from "../charts/PCA2DScatter";
import { fmtPct } from "../../lib/svg";

interface Props { index: IndexDoc | null; }

export function Finding2Scripts({ index }: Props) {
  const [model, setModel] = useState<string>("google/gemma-4-31B");
  const [script, setScript] = useState<string>("devanagari");

  const cell = useMemo(
    () => index ? findCell(index.cells, { model, script, n_max: 100, periods: [2,5,10,100] }) : undefined,
    [index, model, script],
  );

  return (
    <section className="prose-body">
      <h2 className="section-heading">
        It generalises across positional scripts, but only on models trained on them
      </h2>

      <p>
        I rendered every integer 0–99 in eight numeral systems and re-ran
        the analysis. Helix / 9-d PCA ratio at each cell's peak layer:
      </p>

      <table className="article-table mt-4">
        <thead>
          <tr>
            <th>script</th>
            {MODEL_ORDER.map((m) => (
              <th key={m} className="text-right">{MODEL_LABEL[m]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {SCRIPT_ORDER.filter((s) => s !== "babylonian").map((s) => (
            <tr key={s}>
              <td className="font-medium">
                {SCRIPT_LABEL[s]}{" "}
                <span className="ml-1 text-ink-mute font-mono">{SCRIPT_EXAMPLE[s]}</span>
              </td>
              {MODEL_ORDER.map((m) => {
                const c = index ? findCell(index.cells, { model: m, script: s, n_max: 100, periods: [2,5,10,100] }) : undefined;
                return (
                  <td key={m} className="numeric text-right">
                    {fmtPct(c?.helix_over_pca_peak)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <p className="mt-6">
        Three patterns. <strong>Latin is uniform</strong> across all five
        models (≥ 0.75). <strong>Non-positional scripts (Greek, Roman)
        uniformly collapse</strong> the ratio to 0.55–0.65 — a helix with a
        tens loop only buys you anything if the numeral system has a tens
        column. <strong>Non-Latin positional cells split sharply by
        model:</strong> Pythia on Devanagari (0.53) is indistinguishable
        from Pythia on Greek; Gemma-4-31B on Devanagari hits 0.79, matching
        its own Latin. Positional structure is necessary for the helix, but
        training exposure to the specific script is what makes it appear.
      </p>

      <h3 className="section-subheading">Inspect any cell</h3>
      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
        <Picker label="model" value={model} options={MODEL_ORDER} display={(v) => MODEL_LABEL[v]} onChange={setModel} />
        <Picker label="script" value={script} options={[...SCRIPT_ORDER]} display={(v) => `${SCRIPT_LABEL[v]} (${SCRIPT_EXAMPLE[v]})`} onChange={setScript} />
      </div>

      <ChartFrame<CirclesDoc>
        src={cell?.paths.circles}
        minHeight={340}
        caption={<>Per-period circles for <strong>{MODEL_LABEL[model]}</strong> / <strong>{SCRIPT_LABEL[script]}</strong>, plus the linear-axis strip.</>}
      >
        {(d) => <CirclePanel data={d} />}
      </ChartFrame>

      <ChartFrame<Helix3DDoc>
        src={cell?.paths.helix_3d}
        minHeight={480}
        caption="3-D helix for the same cell. If the chosen script lives in this model as a true helix, numbers with matching units-digit stack vertically."
      >
        {(d) => <HelixViewer3D data={d} height={480} />}
      </ChartFrame>

      <ChartFrame<PCA2DDoc>
        src={cell?.paths.pca_2d}
        minHeight={420}
        caption="Basis-free 2D PCA. Latin shows the helix from above. Roman shows a piecewise staircase with jumps at thresholds (4→5, 9→10, 49→50, 89→90). Greek shows tight letter-clusters with no consistent ordering."
      >
        {(d) => <PCA2DScatter data={d} />}
      </ChartFrame>
    </section>
  );
}

function Picker<T extends string>({ label, value, options, display, onChange }: {
  label: string; value: T; options: T[]; display: (v: T) => string; onChange: (v: T) => void;
}) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium uppercase tracking-wider text-ink-mute">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="w-full rounded border border-ink/15 bg-white px-3 py-2 font-sans text-sm"
      >
        {options.map((o) => <option key={o} value={o}>{display(o)}</option>)}
      </select>
    </label>
  );
}
