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
        A random-embedding control reveals that the L=0 Babylonian
        helix is almost entirely mechanical. Replacing every learned
        embedding vector with a random vector of matched scale
        (<span className="font-mono">N(0, 1/√d)</span>) and recomputing
        the L=0 helix R² leaves the base-60 helix essentially unchanged
        across all seven models:
      </p>

      <table className="article-table mt-4">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">learned L=0 R²</th>
            <th className="text-right">random L=0 R²</th>
            <th className="text-right">Δ (learned − random)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Pythia-6.9B</td><td className="numeric text-right">0.569</td><td className="numeric text-right">0.568</td><td className="numeric text-right text-ink-mute">+0.001</td></tr>
          <tr><td>Llama-3.1-8B</td><td className="numeric text-right">0.517</td><td className="numeric text-right">0.513</td><td className="numeric text-right text-ink-mute">+0.003</td></tr>
          <tr><td>Gemma-4-E4B</td><td className="numeric text-right">0.623</td><td className="numeric text-right">0.620</td><td className="numeric text-right text-ink-mute">+0.002</td></tr>
          <tr><td>Gemma-4-31B</td><td className="numeric text-right">0.621</td><td className="numeric text-right">0.621</td><td className="numeric text-right text-ink-mute">−0.001</td></tr>
          <tr><td>OLMo-3-32B</td><td className="numeric text-right">0.545</td><td className="numeric text-right">0.552</td><td className="numeric text-right text-ink-mute">−0.007</td></tr>
          <tr><td>Qwen2.5-7B</td><td className="numeric text-right">0.588</td><td className="numeric text-right">0.590</td><td className="numeric text-right text-ink-mute">−0.002</td></tr>
          <tr><td>Qwen2.5-32B</td><td className="numeric text-right">0.587</td><td className="numeric text-right">0.592</td><td className="numeric text-right text-ink-mute">−0.004</td></tr>
        </tbody>
      </table>

      <p className="mt-4">
        <strong>The learned-vs-random gap is essentially zero on every
        model</strong> (|Δ| ≤ 0.007 across the extended basis). The
        structure arises from additive rendering plus pooling alone:
        the renderer for{" "}
        <span className="font-mono">n=23</span> literally writes two
        ten-wedges and three one-wedges, so
      </p>

      <p className="my-2 ml-6 font-mono text-sm text-ink/80">
        h(23) ≈ (2 · e<sub>𒌋</sub> + 3 · e<sub>𒁹</sub>) / 5
      </p>

      <p>
        and even when{" "}
        <span className="font-mono">e<sub>𒌋</sub>, e<sub>𒁹</sub></span>{" "}
        are random vectors, the result varies smoothly with{" "}
        <span className="font-mono">a</span>, neighbouring numbers
        differ predictably, multiples of 10 form regular shifts, mod-60
        periodicity emerges, and the Fourier basis fits naturally. No
        learned semantics required. The wedge embeddings <em>have not</em>{" "}
        absorbed "𒁹 means one" from cuneiform text; they don't need to
        have.
      </p>

      <p>
        This is the cleanest single result in the paper, and it makes a
        sharper general point: <strong>some apparent neural geometry
        can arise from rendering statistics alone</strong>. When a
        residual stream "encodes" a quantity, the encoding can come from
        arithmetic done in the renderer before the model has read the
        tokens. Provenance analysis needs to extend below L=0 — to the
        rendering pipeline itself. (Whether the transformer reads the
        cuneiform helix downstream is a separate, causal question; see
        the Conclusion.)
      </p>

      <h3 className="section-subheading">Is the effect Babylonian-specific?</h3>

      <p>
        To verify the mechanical reading isn't a general property of
        random embeddings + mean-pooling, I ran the same control across
        every (model, script) cell. The comparison sharpens the story:
      </p>

      <table className="article-table mt-4">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">Latin Δ</th>
            <th className="text-right">Babylonian Δ <span className="text-ink-mute font-normal">(n=600, ext. basis)</span></th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Pythia-6.9B</td><td className="numeric text-right text-accent">+0.10</td><td className="numeric text-right text-ink-mute">+0.001</td></tr>
          <tr><td>Llama-3.1-8B</td><td className="numeric text-right text-accent">+0.15</td><td className="numeric text-right text-ink-mute">+0.003</td></tr>
          <tr><td>Gemma-4-E4B</td><td className="numeric text-right text-accent">+0.12</td><td className="numeric text-right text-ink-mute">+0.002</td></tr>
          <tr><td>Gemma-4-31B</td><td className="numeric text-right text-accent">+0.12</td><td className="numeric text-right text-ink-mute">−0.001</td></tr>
          <tr><td>OLMo-3-32B</td><td className="numeric text-right text-accent">+0.18</td><td className="numeric text-right text-ink-mute">−0.007</td></tr>
          <tr><td>Qwen2.5-7B</td><td className="numeric text-right text-accent">+0.11</td><td className="numeric text-right text-ink-mute">−0.002</td></tr>
          <tr><td>Qwen2.5-32B</td><td className="numeric text-right">+0.03</td><td className="numeric text-right text-ink-mute">−0.004</td></tr>
        </tbody>
      </table>

      <p className="mt-4">
        On <strong>Latin</strong>, every model's learned-vs-random Δ is
        positive and substantial — learned digit embeddings carry numeric
        structure that random embeddings don't. On Pythia/Llama/OLMo the
        random R² is ~0.08 (essentially chance), so the learned helix
        score at L=0 (0.18–0.26) is overwhelmingly learned semantics. On
        <strong> Babylonian</strong>, every model's Δ is ~zero. The
        mechanical effect is real and specific to Babylonian's additive
        within-column rendering.
      </p>

      <p>
        One side observation: Gemma and Qwen have substantially higher
        random R² across <em>every</em> script (~0.4 on positional cells
        vs ~0.08 for Pythia/Llama/OLMo). Their tokenizers split numerals
        more granularly, so even random embeddings, mean-pooled over
        more sub-tokens, produce a vector with non-trivial count
        structure. This doesn't break ρ — Δ between learned and random
        is still meaningfully positive on Latin for Gemma/Qwen — but it
        means the "noise floor" of the protocol is tokenizer-dependent,
        which is itself a useful diagnostic.
      </p>
    </section>
  );
}
