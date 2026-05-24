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
      <h2 className="section-heading">Babylonian: all three failure modes in one cell</h2>

      <p>
        Babylonian cuneiform is positional at base 60 (additive within
        each sexagesimal column). A naive application of the paper
        protocol initially looked like "no helix here" — for two reasons
        that are about <em>measurement</em>, not the model:
      </p>

      <ul className="mt-3 space-y-1 list-disc pl-6">
        <li>
          <strong>Mode 1 (basis bandwidth):</strong> the basis{" "}
          <span className="font-mono">[2, 5, 10, 100]</span> contains no
          T=60 column to fit.
        </li>
        <li>
          <strong>Mode 2 (window must wrap):</strong> the 0..99 input
          range wraps T=60 only once.
        </li>
      </ul>

      <p className="mt-4">
        Fix both — extend the window to <span className="font-mono">n=600</span>{" "}
        (ten wraps of T=60) and add T=60 to the basis — and helix R² jumps
        by <strong>8–16 percentage points</strong> uniformly across all
        seven models, with the largest gain on Qwen-7B (+0.16). The
        basis-free FFT below confirms peaks at multiples of 1/60 once the
        window is wide enough to detect them:
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
        caption={<>Pythia-6.9B on Babylonian, extended to <span className="font-mono">n=600</span>. Auto-detected peaks land at multiples of 1/60.</>}
      >
        {(d) => <FFTSpectrum data={d} />}
      </ChartFrame>

      <p className="mt-6">
        This is the moment to be careful. A naive read would conclude:
        "even a model that has barely seen cuneiform forms a base-60
        helix." But this is also exactly where{" "}
        <strong>mode 3 (tokeniser inheritance)</strong> reasserts itself.
        ρ on Babylonian is <strong>0.82–0.93 on every model</strong> — the
        transformer adds at most ~18 percentage points on top of what the
        wedge-token embeddings supply at L=0. So the +8–16 R² gained from
        fixing modes 1+2 is real signal, but the underlying representation
        is almost entirely tokeniser.
      </p>

      <p>
        The interesting reading: even for a script pre-training has barely
        seen, the tokeniser's per-wedge embeddings absorb enough{" "}
        <em>"<span className="font-mono">𒁹</span> means one,{" "}
        <span className="font-mono">𒌋</span> means ten"</em> from sparse
        text to encode base-60 structure that the trig basis can fit.
        Modern tokenisers, by themselves, are doing more numeric work
        than the literature has been crediting them with. Whether the
        transformer is reading the cuneiform helix for anything
        downstream is a separate question — and one the ρ measurement
        alone can't answer.
      </p>
    </section>
  );
}
