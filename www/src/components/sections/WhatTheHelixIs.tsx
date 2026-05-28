import { useMemo, useState } from "react";
import type { IndexDoc, Helix3DDoc, FourierPC1Doc } from "../../lib/types";
import { findCell, MODEL_ORDER, MODEL_LABEL } from "../../lib/data";
import { ChartFrame } from "../ChartFrame";
import { HelixViewer3D } from "../charts/HelixViewer3D";
import { FFTSpectrum } from "../charts/FFTSpectrum";
import { PC1Scatter } from "../charts/PC1Scatter";
import { M, MM } from "../Math";

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
          <M>{String.raw`B(a) = [a,\, \cos(2\pi a/T),\, \sin(2\pi a/T)]`}</M>{" "}
          for <M>{String.raw`T \in \{2, 5, 10, 100\}`}</M>{" "}
          by least squares. That's <strong>9 features</strong>: one
          linear column (<M>a</M>) plus a cos and a sin for each of
          the four periods (<M>{String.raw`1 + 2\times 4 = 9`}</M>).
          I report <strong>helix R²</strong> alongside the{" "}
          <strong>9-d PCA upper bound</strong> — the variance the{" "}
          <em>best possible</em> 9-dimensional fit could explain
          (PCA's top-9 components; no 9-feature basis can beat it).
          Their ratio (≥ 0.85 on Pythia/Latin) means the trig basis
          isn't just <em>a</em> good 9-dimensional fit — it{" "}
          <em>is</em> the 9-dimensional structure of the residual
          stream. (The number 9 tracks the basis: with a wider basis
          like Babylonian's <M>{String.raw`\{2,5,10,60,100\}`}</M> it
          becomes <M>{String.raw`1 + 2\times 5 = 11`}</M>, and the
          PCA ceiling is computed at the matching dimension. The paper
          writes this generally as the <M>K</M>-d fit.)
        </p>
      </details>

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Deep dive: <em>what the basis B(a) actually contains</em>
        </summary>

        <p>
          For an array of integers <span className="font-mono">a</span> of
          length n, <span className="font-mono">B(a)</span> is a matrix of
          shape <span className="font-mono">(n, 9)</span>. Each row is:
        </p>

        <MM>{String.raw`B(a) \;=\; \big[\, a,\; \cos\tfrac{2\pi a}{2},\, \sin\tfrac{2\pi a}{2},\; \cos\tfrac{2\pi a}{5},\, \sin\tfrac{2\pi a}{5},\; \cos\tfrac{2\pi a}{10},\, \sin\tfrac{2\pi a}{10},\; \cos\tfrac{2\pi a}{100},\, \sin\tfrac{2\pi a}{100} \,\big]`}</MM>

        <p>
          One linear column and four (cos, sin) pairs. The geometry of each
          piece, column by column:
        </p>

        <ul className="mt-3 space-y-2 list-disc pl-6">
          <li>
            <strong>Column 0: <M>a</M></strong> — the "spine." A value
            that grows linearly from 0 to 99. When OLS fits this column
            it picks out the residual-stream direction that best
            correlates with integer magnitude. That direction is the
            number line itself.
          </li>
          <li>
            <strong>Columns 1–2:{" "}
            <M>{String.raw`\cos(2\pi a/2),\, \sin(2\pi a/2)`}</M></strong>{" "}
            — parity. As <M>a</M> steps from 0 to 1 to 2…, the point
            at angle <M>{String.raw`2\pi a/2 = \pi a`}</M> alternates
            between <M>(1, 0)</M> and <M>(-1, 0)</M>. Even and odd
            integers land on opposite sides of a circle.
          </li>
          <li>
            <strong>Columns 3–4:{" "}
            <M>{String.raw`\cos(2\pi a/5),\, \sin(2\pi a/5)`}</M></strong>{" "}
            — mod-5. Five clusters around a circle, one for each
            residue class.
          </li>
          <li>
            <strong>Columns 5–6:{" "}
            <M>{String.raw`\cos(2\pi a/10),\, \sin(2\pi a/10)`}</M></strong>{" "}
            — the units digit. Ten positions around a circle; integers
            ending in the same digit (3, 13, 23, …) land at the same
            angular position. This is the most visually striking
            circle and the one Figure 1 plots in 3D.
          </li>
          <li>
            <strong>Columns 7–8:{" "}
            <M>{String.raw`\cos(2\pi a/100),\, \sin(2\pi a/100)`}</M></strong>{" "}
            — coarse position. Over the input range 0–99 this
            completes exactly one revolution; it's the "big arc" of
            the helix.
          </li>
        </ul>

        <p className="mt-4">
          Why <em>separated</em> cos and sin rather than the complex
          exponential <M>{String.raw`e^{2\pi i a/T}`}</M>? Two
          reasons. (i) The fit is real-valued OLS — splitting into cos and
          sin keeps the linear algebra in reals. (ii) Each (cos, sin) pair
          gives the model a 2D plane to <em>rotate</em> in. Addition becomes
          rotation by the angle of the addend on each clock — the paper's
          "Clock" algorithm.
        </p>

        <p className="mt-3">
          We solve <M>{String.raw`H \approx B(a) \cdot W`}</M> by OLS,
          where <M>{String.raw`H \in \mathbb{R}^{n \times d}`}</M> is
          one row per integer's residual-stream vector (<M>d \approx 4096</M>)
          and <M>{String.raw`W \in \mathbb{R}^{9 \times d}`}</M> contains
          the nine directions in residual space (one per basis feature).
          The columns of W tell us <em>where</em> each clock and the
          number-line direction live in residual space. Figure 1
          extracts the three columns{" "}
          <M>{String.raw`(\cos(2\pi a/10),\, \sin(2\pi a/10),\, a)`}</M>,
          QR-orthonormalises them, projects H onto the resulting clean
          3D frame, and renders the helix.
        </p>

        <p className="mt-3">
          The variance-weighted R² we report puts more weight on the hidden
          dimensions that have non-trivial variance in <span className="font-mono">a</span>{" "}
          — so the score isn't dominated by the many residual-stream
          dimensions that are nearly constant across integers (e.g. tokens
          for "language", "decimal", etc.).
        </p>
      </details>

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Deep dive: <em>what "9-dimensional" means, and the quality ratio</em>
        </summary>

        <p>
          The number 9 keeps showing up — "9 features," "9-d PCA,"
          "the 9-dimensional structure." It's worth pinning down
          exactly what it counts and why the comparison is built the
          way it is.
        </p>

        <p className="mt-3">
          <strong>Where 9 comes from.</strong> Count the columns of{" "}
          <M>B(a)</M>:
        </p>
        <ul className="mt-2 list-disc pl-6 space-y-1">
          <li>1 linear feature: <M>a</M> (the number-line spine).</li>
          <li>
            4 periods, and each period contributes <em>two</em>{" "}
            features — a <M>\cos</M> and a <M>\sin</M>:{" "}
            <M>{String.raw`4 \times 2 = 8`}</M>.
          </li>
          <li>Total: <M>{String.raw`1 + 8 = 9`}</M>.</li>
        </ul>
        <p className="mt-2">
          So the general count is{" "}
          <M>{String.raw`K = 1 + 2\times(\text{number of periods})`}</M>.
          The paper writes "K-d" because <strong>K changes with the
          basis</strong>:
        </p>

        <table className="article-table mt-3 text-[0.95rem]">
          <thead>
            <tr>
              <th>basis</th>
              <th className="text-right">periods</th>
              <th className="text-right">K</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="font-mono">{"{2, 5, 10, 100}"} (paper default)</td><td className="numeric text-right">4</td><td className="numeric text-right">9</td></tr>
            <tr><td className="font-mono">{"{2, 5, 10, 60, 100}"} (Babylonian)</td><td className="numeric text-right">5</td><td className="numeric text-right">11</td></tr>
            <tr><td className="font-mono">{"{2, 4, 8, 16, 32, 64}"} (binary)</td><td className="numeric text-right">6</td><td className="numeric text-right">13</td></tr>
            <tr><td className="font-mono">{"{16, 32, 64, 256}"} (hex)</td><td className="numeric text-right">4</td><td className="numeric text-right">9</td></tr>
          </tbody>
        </table>
        <p className="mt-2">
          "9-d" is just the paper-default case of "K-d."
        </p>

        <p className="mt-4">
          <strong>What "K-dimensional" means.</strong> When we fit the
          9-feature basis, we're projecting the residual stream onto a
          9-dimensional subspace — the 9 directions in{" "}
          <M>W</M>, one per feature. Helix R² is how much of the
          integer-variation those 9 directions explain. Fitting a
          basis with more features means projecting onto a
          higher-dimensional subspace.
        </p>

        <p className="mt-4">
          <strong>Why the PCA ceiling, and why at the same K.</strong>{" "}
          Here's the subtlety. <em>More dimensions always fit more
          variance.</em> If I compared a 9-feature trig basis against
          a 50-dimensional PCA, the 50-d would win every time — not
          because the trig basis is bad, but because it has more knobs.
          That's not a fair test of "is the trig basis good?"
        </p>

        <p className="mt-3">
          So we compare against the <em>best possible 9-dimensional
          fit</em>. PCA gives exactly that: a classic result (the
          Eckart–Young theorem) says the top-9 principal components are
          the mathematically optimal 9-dimensional linear
          approximation of any matrix. Nothing 9-dimensional beats it.
          So we always have
        </p>

        <MM>{String.raw`R^2_{\text{helix}} \;\le\; R^2_{\text{9-d PCA}}`}</MM>

        <p>
          and the ratio{" "}
          <M>{String.raw`R^2_{\text{helix}} / R^2_{\text{9-d PCA}}`}</M>{" "}
          measures <strong>how close the trig basis gets to the
          best-possible fit at the same dimensional budget</strong>.
          A ratio of 0.85 means the helix captures 85% of what{" "}
          <em>any</em> 9-dimensional structure could capture from this
          activation matrix. (For a non-default basis we compute the
          PCA ceiling at the matching K — so Babylonian's 11-feature
          fit is compared against the 11-d PCA ceiling, not the 9-d
          one.)
        </p>

        <p className="mt-4">
          <strong>What "the 9-dimensional structure of the residual
          stream" means.</strong> This is the phrase most worth
          unpacking, because it's easy to misread. It does{" "}
          <em>not</em> mean the residual stream is 9-dimensional — it's{" "}
          <M>d \approx 4096</M>-dimensional. It means:
        </p>

        <p className="my-3 ml-4 border-l-2 border-accent/40 pl-4 text-ink">
          The part of the 4096-dimensional residual stream that{" "}
          <em>varies with the integer</em> is approximately
          9-dimensional, and the helix basis <em>is</em> those 9
          dimensions (to ~85%).
        </p>

        <p>
          The other ~4087 dimensions either don't change as{" "}
          <M>a</M> goes 0→99, or vary in ways the top-9 PCA already
          absorbs. So when we say "the helix is the 9-dimensional
          structure," we're answering the question: <em>what are the
          ~9 directions in which this model's residual stream moves
          as the integer changes?</em> The answer is the helix — the
          number line plus the four clocks. That's a strong claim:
          it's not that a helix is <em>one</em> pattern you can find
          in the activations, it's that the helix is essentially{" "}
          <em>all</em> of the integer-dependent structure there is to
          find at this dimensionality.
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
