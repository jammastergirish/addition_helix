import { useMemo, useState } from "react";
import type { IndexDoc, Helix3DDoc, FourierPC1Doc } from "../../lib/types";
import { findCell, MODEL_ORDER, MODEL_LABEL } from "../../lib/data";
import { ChartFrame } from "../ChartFrame";
import { HelixViewer3D } from "../charts/HelixViewer3D";
import { FFTSpectrum } from "../charts/FFTSpectrum";
import { PC1Scatter } from "../charts/PC1Scatter";

interface Props { index: IndexDoc | null; }

const DEFAULT_MODEL = "EleutherAI/pythia-6.9b";

export function WhatTheHelixIs({ index }: Props) {
  const [model, setModel] = useState<string>(DEFAULT_MODEL);

  const cell = useMemo(
    () => index ? findCell(index.cells, {
      model, script: "latin", n_max: 100, periods: [2, 5, 10, 100],
    }) : undefined,
    [index, model],
  );

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

      <ModelPills value={model} onChange={setModel} />

      <ChartFrame<Helix3DDoc>
        src={cell?.paths.helix_3d}
        minHeight={500}
        caption={
          cell ? (
            <>
              <strong>{MODEL_LABEL[model]}</strong> on Latin digits 0–99,
              read at layer {cell.peak_layer} of {cell.n_layers} (the layer
              where helix R² peaks). 3-D projection onto an orthonormal
              frame of (u<sub>cos(2π·a/10)</sub>, u<sub>sin(2π·a/10)</sub>,
              u<sub>lin</sub>). Drag to rotate; scroll to zoom.
            </>
          ) : (
            <>3-D helix. Drag to rotate; scroll to zoom.</>
          )
        }
      >
        {(data) => <HelixViewer3D data={data} height={500} />}
      </ChartFrame>

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ How I measure it
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
          Finally I fit a trig basis{" "}
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

/**
 * Pills that switch which model's helix is shown above. Latin/peak-layer
 * is held constant so the comparison is apples-to-apples: same script,
 * each model read at its own helix-R² peak.
 */
function ModelPills({ value, onChange }: { value: string; onChange: (m: string) => void }) {
  return (
    <div className="mt-4 mb-1 flex flex-wrap items-center gap-2 text-xs">
      <span className="text-ink-mute uppercase tracking-wider font-medium">model:</span>
      {MODEL_ORDER.map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={[
            "rounded-full border px-3 py-1 font-medium tracking-wide transition",
            value === m
              ? "border-accent bg-accent/10 text-accent"
              : "border-ink/15 text-ink-mute hover:border-ink/30 hover:text-ink",
          ].join(" ")}
        >
          {MODEL_LABEL[m]}
        </button>
      ))}
    </div>
  );
}
