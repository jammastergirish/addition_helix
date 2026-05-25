import type { IndexDoc } from "../../lib/types";
import { findCell, MODEL_ORDER, MODEL_LABEL } from "../../lib/data";
import { fmtPct } from "../../lib/svg";

interface Props { index: IndexDoc | null; }

/**
 * Binary and hexadecimal as parallel case studies in mode 1 (basis
 * bandwidth). Sits right after Babylonian. The structural parallel:
 *
 *   - Babylonian: native period T=60; paper basis catches none of it.
 *     Fix: n=600, basis +T=60. Gain ≈ +0.10 R² on every model.
 *   - Binary:     native periods 2,4,8,…,64; paper catches only T=2.
 *     Fix: n=1024, basis [2,4,8,16,32,64]. Gain ≈ +0.10–0.37.
 *   - Hex:        native periods 16,32,64,256; paper catches almost
 *     none. Fix: n=1024, basis [16,32,64,256]. Gain ≈ +0.19–0.29.
 *
 * The twist: hex is *the second-cleanest depth-built case* in the
 * matrix after Latin. Pythia/Llama/GPT-J/OLMo all have ρ ≈ 0.29–0.35
 * on the hex-native basis — depth is constructing the helix, not
 * inheriting it. The natural guess is hex exposure during pre-training
 * (code, addresses, hashes) but I don't have data to test it directly.
 */
export function FindingBases({ index }: Props) {
  // Helper to pull a specific (model, script, n_max, periods) cell.
  const get = (model: string, script: string, n_max: number, periods: number[]) =>
    index ? findCell(index.cells, { model, script, n_max, periods }) : undefined;

  return (
    <section className="prose-body">
      <h2 className="section-heading">
        Binary and hexadecimal: two more mode-1 case studies
      </h2>

      <p>
        Babylonian's story — paper basis misses T=60, fixing the basis
        and widening the window recovers real structure — has direct
        parallels in any base that isn't 10. Binary and hexadecimal are
        the obvious tests:
      </p>

      <ul className="mt-3 space-y-1 list-disc pl-6">
        <li>
          <strong>Binary</strong>: natural periods 2, 4, 8, 16, 32, 64.
          Paper basis catches only T = 2.
        </li>
        <li>
          <strong>Hexadecimal</strong>: natural periods 16, 32, 64, 128,
          256. Paper basis catches almost nothing relevant.
        </li>
      </ul>

      <p className="mt-4">
        Run each at <span className="font-mono">n=1024</span> (16+ wraps
        of any period below 64) with a basis tuned to the script's actual
        base:
      </p>

      <h3 className="section-subheading">Binary, paper basis vs binary-native</h3>

      <table className="article-table mt-3">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">paper R² <span className="font-normal text-ink-mute">[2,5,10,100]</span></th>
            <th className="text-right">native R² <span className="font-normal text-ink-mute">[2,4,8,16,32,64]</span></th>
            <th className="text-right">Δ</th>
            <th className="text-right">ρ (native)</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_ORDER.map((m) => {
            const base = get(m, "binary", 1024, [2,5,10,100]);
            const ext  = get(m, "binary", 1024, [2,4,8,16,32,64]);
            const d = (base?.helix_r2_peak != null && ext?.helix_r2_peak != null)
              ? ext.helix_r2_peak - base.helix_r2_peak : null;
            return (
              <tr key={m}>
                <td className="font-medium">{MODEL_LABEL[m]}</td>
                <td className="numeric text-right">{fmtPct(base?.helix_r2_peak)}</td>
                <td className="numeric text-right">{fmtPct(ext?.helix_r2_peak)}</td>
                <td className="numeric text-right text-accent">
                  {d == null ? "—" : `+${d.toFixed(2)}`}
                </td>
                <td className="numeric text-right">{fmtPct(ext?.rho)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p className="mt-3">
        <strong>+0.10 to +0.37 points of R² across all 8 models</strong>{" "}
        from fixing the basis. The biggest gains are on Qwen and Gemma
        (+0.29 to +0.37) — models with tokenizers that split numbers
        into smaller sub-tokens, making the per-bit count structure
        more visible after mean-pooling.
      </p>

      <p className="mt-3">
        But the random-embedding control reveals that{" "}
        <strong>binary's L=0 helix is mechanical on every model</strong>{" "}
        — same story as Babylonian. The learned-vs-random gap on the
        binary-native basis is{" "}
        <strong>|Δ R²| ≤ 0.03 on every model</strong> (often negative;
        learned and random are essentially identical):
      </p>

      <table className="article-table mt-3">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">learned L=0 R²</th>
            <th className="text-right">random L=0 R²</th>
            <th className="text-right">Δ</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Pythia-6.9B</td><td className="numeric text-right">0.150</td><td className="numeric text-right">0.126</td><td className="numeric text-right text-ink-mute">+0.025</td></tr>
          <tr><td>GPT-J-6B</td><td className="numeric text-right">0.175</td><td className="numeric text-right">0.147</td><td className="numeric text-right text-ink-mute">+0.028</td></tr>
          <tr><td>Llama-3.1-8B</td><td className="numeric text-right">0.214</td><td className="numeric text-right">0.214</td><td className="numeric text-right text-ink-mute">+0.000</td></tr>
          <tr><td>Gemma-4-E4B</td><td className="numeric text-right">0.625</td><td className="numeric text-right">0.647</td><td className="numeric text-right text-ink-mute">−0.022</td></tr>
          <tr><td>Gemma-4-31B</td><td className="numeric text-right">0.629</td><td className="numeric text-right">0.646</td><td className="numeric text-right text-ink-mute">−0.018</td></tr>
          <tr><td>OLMo-3-32B</td><td className="numeric text-right">0.207</td><td className="numeric text-right">0.217</td><td className="numeric text-right text-ink-mute">−0.010</td></tr>
          <tr><td>Qwen2.5-7B</td><td className="numeric text-right">0.644</td><td className="numeric text-right">0.646</td><td className="numeric text-right text-ink-mute">−0.001</td></tr>
          <tr><td>Qwen2.5-32B</td><td className="numeric text-right">0.649</td><td className="numeric text-right">0.646</td><td className="numeric text-right text-ink-mute">+0.002</td></tr>
        </tbody>
      </table>

      <p className="mt-4">
        The mechanism here is rendering + tokenization + pooling
        statistics — not pure bit-count, which alone would only give
        Hamming weight and couldn't separate many distinct numbers.
        Several rendering statistics carry information about{" "}
        <em>n</em>: how many tokens the binary string occupies (length
        scales with log<sub>2</sub> n), how the tokenizer chunks the
        bit string (multi-bit BPE merges may split high vs low
        positions differently), where the boundary tokens fall, and
        any positional embedding the model adds at layer 0. Combined,
        these recover enough structure for the trig basis to fit —
        with random or learned embeddings, similarly.
      </p>

      <p>
        The ρ column on Pythia/Llama/OLMo/GPT-J (0.62–0.77) therefore
        doesn't mean those models learned binary number-sense. It means
        depth is doing additional work <em>on top of</em> a mechanical
        L=0 helix — and the CKA data (Finding 5) confirms it: those
        cells have moderate CKA (0.55–0.71), so depth amplifies/rotates
        the mechanical L=0 rather than rebuilding from scratch.
      </p>

      <h3 className="section-subheading">Hexadecimal — the surprise</h3>

      <table className="article-table mt-3">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">paper R² <span className="font-normal text-ink-mute">[2,5,10,100]</span></th>
            <th className="text-right">native R² <span className="font-normal text-ink-mute">[16,32,64,256]</span></th>
            <th className="text-right">Δ</th>
            <th className="text-right">ρ (native)</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_ORDER.map((m) => {
            const base = get(m, "hexadecimal", 1024, [2,5,10,100]);
            const ext  = get(m, "hexadecimal", 1024, [16,32,64,256]);
            const d = (base?.helix_r2_peak != null && ext?.helix_r2_peak != null)
              ? ext.helix_r2_peak - base.helix_r2_peak : null;
            return (
              <tr key={m}>
                <td className="font-medium">{MODEL_LABEL[m]}</td>
                <td className="numeric text-right">{fmtPct(base?.helix_r2_peak)}</td>
                <td className="numeric text-right">{fmtPct(ext?.helix_r2_peak)}</td>
                <td className="numeric text-right text-accent">
                  {d == null ? "—" : `+${d.toFixed(2)}`}
                </td>
                <td className="numeric text-right font-medium">{fmtPct(ext?.rho)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p className="mt-3">
        Δ R² of <strong>+0.19 to +0.29</strong> — even larger than
        Babylonian's +0.08–+0.16. But look at the ρ column:
      </p>

      <p className="mt-3 rounded border-l-4 border-accent bg-accent/5 px-4 py-3 text-base text-ink">
        <strong>On Pythia (ρ = 0.29), Llama (0.30), GPT-J (0.35), and
        OLMo (0.35), hex is as depth-built as Latin.</strong> Pythia/Latin
        and Llama/Latin sit at ρ = 0.34 and 0.39 — the original paper's
        two clean cells. Hex on those same models is in the same regime
        (sometimes lower). The hexadecimal helix is genuinely constructed
        by the transformer, not inherited.
      </p>

      <p>
        This was unexpected. The natural hypothesis is exposure to hex
        in pre-training — most LLMs see hex in code, addresses, hashes,
        RGB colours — but the data here doesn't directly test it.
        Whatever the cause, the model is treating hex digits the way it
        treats Latin digits: as indices into a value space, with
        layer-by-layer construction of the same kind of helix.
      </p>

      <p>
        Gemma's hex picture is more protocol-dependent than the other
        four models'. On the <em>paper-default</em> protocol, Gemma-31B
        sits at ρ = 0.34, CKA = 0.67 and Gemma-E4B at ρ = 0.43, CKA =
        0.79 — both look like depth amplification of a real L=0 helix.
        On the <em>hex-native</em> protocol used by the table above
        (n=1024, basis [16,32,64,256]), Gemma's ρ rises to 0.54 and 0.63
        — closer to the inherited / depth-amplified border than to the
        depth-built regime where Pythia/Llama/GPT-J/OLMo land. Either
        way, Gemma is using hex more like Latin than like the
        all-Gemma-scripts rebuild pattern of Finding 5.
      </p>

      <h3 className="section-subheading">The random-embedding control confirms it</h3>

      <p>
        Running the random-embedding control on hex with the hex-native
        basis: the learned-vs-random gap is{" "}
        <strong>positive on every model</strong> — small on
        Llama/OLMo, sizeable on Gemma:
      </p>

      <table className="article-table mt-3">
        <thead>
          <tr>
            <th>model</th>
            <th className="text-right">learned L=0 R²</th>
            <th className="text-right">random L=0 R²</th>
            <th className="text-right">Δ</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Pythia-6.9B</td><td className="numeric text-right">0.085</td><td className="numeric text-right">0.036</td><td className="numeric text-right text-accent">+0.049</td></tr>
          <tr><td>GPT-J-6B</td><td className="numeric text-right">0.095</td><td className="numeric text-right">0.037</td><td className="numeric text-right text-accent">+0.058</td></tr>
          <tr><td>Llama-3.1-8B</td><td className="numeric text-right">0.102</td><td className="numeric text-right">0.085</td><td className="numeric text-right text-accent">+0.017</td></tr>
          <tr className="font-medium"><td>Gemma-4-E4B</td><td className="numeric text-right">0.229</td><td className="numeric text-right">0.133</td><td className="numeric text-right text-accent">+0.096</td></tr>
          <tr className="font-medium"><td>Gemma-4-31B</td><td className="numeric text-right">0.202</td><td className="numeric text-right">0.134</td><td className="numeric text-right text-accent">+0.068</td></tr>
          <tr><td>OLMo-3-32B</td><td className="numeric text-right">0.091</td><td className="numeric text-right">0.085</td><td className="numeric text-right text-accent">+0.006</td></tr>
          <tr><td>Qwen2.5-7B</td><td className="numeric text-right">0.173</td><td className="numeric text-right">0.135</td><td className="numeric text-right text-accent">+0.038</td></tr>
          <tr><td>Qwen2.5-32B</td><td className="numeric text-right">0.162</td><td className="numeric text-right">0.133</td><td className="numeric text-right text-accent">+0.029</td></tr>
        </tbody>
      </table>

      <p className="mt-4">
        Hex digit embeddings carry meaningful structure that random
        embeddings don't. Combined with the high peak helix R² and
        low ρ on Pythia/Llama/GPT-J/OLMo, the diagnostic kit gives a
        fully consistent <em>depth-built</em> story for hex on those
        models — with modest learned-embedding contribution at L=0 and
        a substantial depth-driven amplification on top.
      </p>

      <h3 className="section-subheading">Three case studies, three different mode-1 stories</h3>

      <p>
        Babylonian, binary, and hexadecimal each break the paper protocol
        for the same reason (wrong basis) but tell different mechanistic
        stories once the basis is fixed:
      </p>

      <ul className="mt-3 space-y-2 list-disc pl-6">
        <li>
          <strong>Babylonian</strong> — ρ stays high (0.82–0.93), random
          ≈ learned at L=0. Mechanical: structure is in rendering +
          pooling alone.
        </li>
        <li>
          <strong>Binary</strong> — <em>also</em> mechanical (|Δ_random|
          ≤ 0.03 on every model). The bit-count renderer composes with
          mean-pooling to give a per-<em>n</em> linear vector with or
          without learned embeddings. Pythia/Llama/OLMo/GPT-J's
          moderately-low ρ (0.44–0.77) reflects depth doing additional
          work on top of a mechanical L=0, not learned bit semantics.
        </li>
        <li>
          <strong>Hex</strong> — the depth-built case. Learned hex
          embeddings carry meaningful structure (Δ_random {">"} 0 on every
          model), and depth roughly doubles the score on
          Pythia/Llama/GPT-J/OLMo (ρ = 0.29–0.35). Hex exposure during
          pre-training is the natural hypothesis, but unverified here.
        </li>
      </ul>

      <p className="mt-4">
        The point is not "every base is a helix." It's that{" "}
        <em>each base demands its own protocol fix</em>, and once you
        apply the fix, ρ tells you immediately whether what you find is
        learned, mechanical, or somewhere between.
      </p>
    </section>
  );
}
