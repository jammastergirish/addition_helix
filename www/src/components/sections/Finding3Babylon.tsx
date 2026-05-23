import type { IndexDoc, FourierPC1Doc } from "../../lib/types";
import { findCell, MODEL_ORDER, MODEL_LABEL } from "../../lib/data";
import { ChartFrame } from "../ChartFrame";
import { FFTSpectrum } from "../charts/FFTSpectrum";
import { fmtPct } from "../../lib/svg";

interface Props { index: IndexDoc | null; }

export function Finding3Babylon({ index }: Props) {
  const fftCell = index ? findCell(index.cells, {
    model: "EleutherAI/pythia-6.9b", script: "babylonian",
    n_max: 600, periods: [2, 5, 10, 100],
  }) : undefined;

  return (
    <section className="prose-body">
      <h2 className="section-heading">Even Babylonian cuneiform encodes base 60</h2>

      <p>
        Babylonian is positional at base 60 (additive within each
        sexagesimal column). The standard 0..99 measurement protocol
        initially looked like it said "no helix" — but for two reasons that
        are about the measurement, not the model: the input range wraps T=60
        only once, and the paper's basis{" "}
        <span className="font-mono">[2, 5, 10, 100]</span> contains no T=60
        column to fit. Extending to <span className="font-mono">n=600</span>{" "}
        (ten wraps) and adding T=60 to the basis recovers 8–13 percentage
        points of helix R² <em>uniformly</em>, on every model:
      </p>

      <table className="article-table mt-4">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">paper basis</th>
            <th className="text-right">+ T=60</th>
            <th className="text-right">Δ</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_ORDER.map((m) => {
            const c1 = index ? findCell(index.cells, { model: m, script: "babylonian", n_max: 600, periods: [2,5,10,100] }) : undefined;
            const c2 = index ? findCell(index.cells, { model: m, script: "babylonian", n_max: 600, periods: [2,5,10,60,100] }) : undefined;
            const delta = (c1?.helix_r2_peak != null && c2?.helix_r2_peak != null)
              ? c2.helix_r2_peak - c1.helix_r2_peak
              : null;
            return (
              <tr key={m}>
                <td className="font-medium">{MODEL_LABEL[m]}</td>
                <td className="numeric text-right">{fmtPct(c1?.helix_r2_peak)}</td>
                <td className="numeric text-right">{fmtPct(c2?.helix_r2_peak)}</td>
                <td className="numeric text-right text-accent">
                  {delta == null ? "—" : `+${delta.toFixed(2)}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <ChartFrame<FourierPC1Doc>
        src={fftCell?.paths.fourier_pc1}
        minHeight={300}
        caption={<>Pythia-6.9B on Babylonian, extended to <span className="font-mono">n=600</span>. Auto-detected peaks land at multiples of 1/60 — base-60 structure visible in the residual stream of a model that has barely seen cuneiform.</>}
      >
        {(d) => <FFTSpectrum data={d} />}
      </ChartFrame>
    </section>
  );
}
