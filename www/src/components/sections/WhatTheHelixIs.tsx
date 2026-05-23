import type { IndexDoc, Helix3DDoc, FourierPC1Doc } from "../../lib/types";
import { findCell } from "../../lib/data";
import { ChartFrame } from "../ChartFrame";
import { HelixViewer3D } from "../charts/HelixViewer3D";
import { FFTSpectrum } from "../charts/FFTSpectrum";
import { PC1Scatter } from "../charts/PC1Scatter";

interface Props { index: IndexDoc | null; }

const PYTHIA_LATIN = {
  model: "EleutherAI/pythia-6.9b", script: "latin",
  n_max: 100, periods: [2, 5, 10, 100],
};

export function WhatTheHelixIs({ index }: Props) {
  const cell = index ? findCell(index.cells, PYTHIA_LATIN) : undefined;

  return (
    <section className="prose-body">
      <h2 className="section-heading">The helix</h2>
      <p>
        For an integer <span className="font-mono">a ∈ [0, 99]</span>, the
        residual stream encodes <span className="font-mono">a</span> as a 3-D
        spiral: a linear "number-line" axis lifts the point as{" "}
        <span className="font-mono">a</span> grows, while circular loops at
        periods <span className="font-mono">T ∈ {"{"}2, 5, 10, 100{"}"}</span>{" "}
        wrap it around. Period 10 gives the units digit (numbers ending in 7
        stack vertically); period 100 gives coarse magnitude; 2 gives parity;
        5 gives <span className="font-mono">a mod 5</span>. Addition becomes
        rotating around the loops — the paper's "Clock" algorithm.
      </p>

      <ChartFrame<Helix3DDoc>
        src={cell?.paths.helix_3d}
        minHeight={500}
        caption={<>Pythia-6.9B, layer 32 of 32. 3-D projection onto an orthonormal frame of (u<sub>cos(2π·a/10)</sub>, u<sub>sin(2π·a/10)</sub>, u<sub>lin</sub>). Drag to rotate; scroll to zoom.</>}
      >
        {(data) => <HelixViewer3D data={data} height={500} />}
      </ChartFrame>

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ How we measure it
        </summary>
        <p>
          Two basis-free diagnostics. <strong>FFT each hidden dimension:</strong>{" "}
          treat the 100 values{" "}
          <span className="font-mono">h(0)[j], h(1)[j], …, h(99)[j]</span> as a
          signal indexed by <span className="font-mono">a</span>; the FFT picks out
          its periodic content. A peak at frequency{" "}
          <span className="font-mono">1/T</span> means "this dimension oscillates
          with period <span className="font-mono">T</span>." Averaging
          magnitudes across all ~4096 hidden dims washes out noise; periods
          many dims agree on survive.
        </p>

        <ChartFrame<FourierPC1Doc>
          src={cell?.paths.fourier_pc1}
          minHeight={280}
          caption="Averaged FFT magnitude. Dashed grey lines: reference periods T = 2, 5, 10, 100. Orange dots: top-5 auto-detected peaks. They land on the reference lines."
        >
          {(d) => <FFTSpectrum data={d} />}
        </ChartFrame>

        <p className="mt-3">
          <strong>PC1 vs <span className="font-mono">a</span>:</strong> the
          first principal component is the direction of maximum variance.
          Plotting it against <span className="font-mono">a</span> tests
          whether the model lays numbers out as a number line. R² ≈ 0.95 on
          Pythia/Latin.
        </p>

        <ChartFrame<FourierPC1Doc>
          src={cell?.paths.fourier_pc1}
          minHeight={280}
          caption="PC1 of the residual stream against integer a, with the linear best-fit and R²."
        >
          {(d) => <PC1Scatter data={d} />}
        </ChartFrame>

        <p className="mt-3">
          Finally we fit a trig basis{" "}
          <span className="font-mono">B(a) = [a, cos(2πa/T), sin(2πa/T)]</span>{" "}
          for <span className="font-mono">T ∈ {"{"}2, 5, 10, 100{"}"}</span>{" "}
          (9 features) by least squares, and report <strong>helix R²</strong>{" "}
          alongside the <strong>9-d PCA upper bound</strong>. Their ratio
          (≥ 0.85 on Pythia/Latin) means the trig basis isn't just{" "}
          <em>a</em> good 9-D fit — it <em>is</em> the 9-D structure of the
          residual stream.
        </p>
      </details>
    </section>
  );
}
