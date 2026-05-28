import type { IndexDoc, FourierPC1Doc } from "../../lib/types";
import { findCell, MODEL_ORDER, MODEL_LABEL } from "../../lib/data";
import { ChartFrame } from "../ChartFrame";
import { FFTSpectrum } from "../charts/FFTSpectrum";
import { fmtPct } from "../../lib/svg";
import { M, MM } from "../Math";

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
        eight models, with the largest gain on Qwen-7B (+0.16). The
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

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Deep dive: <em>basis bandwidth and window wrap, separately</em>
        </summary>

        <p>
          The two failure modes that need fixing here aren't the same
          thing. They interact in a way that's worth pulling apart.
        </p>

        <p className="mt-3">
          <strong>Basis bandwidth.</strong> The trig basis can only fit
          periods it explicitly contains. Babylonian uses base 60. If
          the model encodes that structure at all, the natural period
          is T=60. But T=60 is not in <span className="font-mono">{"{"}2, 5, 10, 100{"}"}</span>,
          so the basis has no cos/sin column to fit it. Even if the
          model perfectly encodes base-60 structure, the OLS fit reports
          low R² because the basis doesn't have the right columns.
          The model has signal; the instrument can't see it. Like
          trying to detect a 1 kHz tone with a microphone that only
          captures 5–20 kHz — the tone is there, the microphone reports
          "no signal," and that's a fact about the microphone, not the
          room.
        </p>

        <p className="mt-3">
          <strong>Window wrap.</strong> Even if we put T=60 in the
          basis, n=100 wraps T=60 only <em>once</em> (around a=60).
          One wrap is not enough to identify a periodic component.
        </p>

        <p className="mt-3">
          The Fourier intuition: the FFT of a length-100 signal at
          frequency f = 1/60 doesn't form a clean single bin because
          100 isn't an integer multiple of 60. The energy of a true
          T=60 component spreads across multiple frequency bins
          ("spectral leakage"), and the peak at 1/60 becomes
          indistinguishable from noise. With 10 cycles in 600 samples
          (n=600), the leakage is small and the peak is sharp — which
          is exactly what the FFT panel above shows.
        </p>

        <p className="mt-3">
          The regression intuition: with one wrap of T=60 over 100
          samples, the columns{" "}
          <span className="font-mono">cos(2πa/60)</span> and{" "}
          <span className="font-mono">sin(2πa/60)</span> are highly
          correlated with all sorts of other columns and with each
          other. OLS can fit "something" but the result is unstable
          and not attributable to T=60 structure specifically. With
          ten wraps the cos/sin columns are well-separated from each
          other and from auxiliary features; the solver gets a clean,
          well-identified fit.
        </p>

        <p className="mt-3">
          Rule of thumb: you need the input range to wrap any
          candidate period at least 8–10 times for clean
          identification.
        </p>

        <p className="mt-3">
          <strong>Both have to be fixed.</strong> For Babylonian the
          paper protocol (n=100, basis [2,5,10,100]) fails both
          tests. We can pull them apart by running three passes:
        </p>

        <table className="article-table mt-3 text-[0.95rem]">
          <thead>
            <tr>
              <th>pass</th>
              <th>n_max</th>
              <th>basis</th>
              <th>what's tested</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td className="numeric">100</td>
              <td className="font-mono">{"{2,5,10,100}"}</td>
              <td>paper default — score stays low</td>
            </tr>
            <tr>
              <td>2</td>
              <td className="numeric">600</td>
              <td className="font-mono">{"{2,5,10,100}"}</td>
              <td>wider window only — score <em>still</em> low</td>
            </tr>
            <tr>
              <td>3</td>
              <td className="numeric">600</td>
              <td className="font-mono">{"{2,5,10,60,100}"}</td>
              <td>window + basis fixed — score jumps</td>
            </tr>
          </tbody>
        </table>

        <p className="mt-3">
          Pass 2 is informative on its own: it tells us fixing only
          the window isn't enough — the basis was also wrong. Pass 3
          then shows the structure was actually there in the model;
          the original "no helix on Babylonian" reading at paper
          defaults was a measurement artifact, not a fact about what
          the model encodes.
        </p>

        <p className="mt-3">
          The contrast between pass 2 (window-only fix, helix R²
          stays low) and pass 3 (window + basis fix, helix R² jumps
          8–16pp uniformly across all 8 models) is the
          basis-bandwidth diagnostic with numbers attached. And the
          basis-free FFT in pass 3 shows clean peaks at multiples
          of 1/60, confirming that the model encodes period-60
          structure that the paper basis could never have caught.
        </p>

        <p className="mt-3">
          <strong>Beyond non-decimal bases.</strong> These failure
          modes aren't specific to Babylonian. Any paper looking at
          a representation through "fit a trig basis and report R²"
          can fall into them. The check is mechanical: vary the
          basis, vary the window, see if the score changes. If it
          does, the original protocol was hiding real structure.
        </p>
      </details>

      <p className="mt-6">
        This is the moment to be careful. A naive read would conclude:
        "even a model that has barely seen cuneiform forms a base-60
        helix." But this is also exactly where{" "}
        <strong>mode 3 (tokenizer inheritance)</strong> reasserts itself.
        ρ on Babylonian is <strong>0.82–0.93 on every model</strong> — the
        transformer adds at most ~18 percentage points on top of what the
        wedge-token embeddings supply at L=0. So the +8–16 R² gained from
        fixing modes 1+2 is real signal, but the underlying representation
        is almost entirely tokenizer.
      </p>

      <p>
        A random-embedding control reveals that the L=0 Babylonian
        helix is almost entirely mechanical. Replacing every learned
        embedding vector with a random vector of matched scale
        (<span className="font-mono">N(0, 1/√d)</span>) and recomputing
        the L=0 helix R² leaves the base-60 helix essentially unchanged
        across all eight models:
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
          <tr><td>GPT-J-6B</td><td className="numeric text-right">0.566</td><td className="numeric text-right">0.567</td><td className="numeric text-right text-ink-mute">−0.002</td></tr>
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
        measurement provides no evidence that the wedge embeddings
        learned "𒁹 means one" — the same score appears with random
        embeddings. The structure arises mechanically from rendering,
        tokenization, and pooling statistics. For{" "}
        <span className="font-mono">n=23</span> the renderer writes two
        ten-wedges and three one-wedges, so:
      </p>

      <MM>{String.raw`h(23) \;\approx\; \frac{2\, e_{\text{𒌋}} \;+\; 3\, e_{\text{𒁹}}}{5}`}</MM>

      <p>
        and even when{" "}
        <span className="font-mono">e<sub>𒌋</sub>, e<sub>𒁹</sub></span>{" "}
        are random vectors, the result varies smoothly with{" "}
        <span className="font-mono">a</span>, neighbouring numbers
        differ predictably, and the Fourier basis fits the per-symbol
        counts. (The story is a little more nuanced above n=59, where
        wedges appear in multiple sexagesimal columns and pure
        bag-of-wedges loses column position — but the rendering-pooling
        statistics still carry enough structure for the helix score to
        reproduce.)
      </p>

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Deep dive: <em>the four-step chain for n=23, Babylonian on Pythia</em>
        </summary>

        <p>
          Walking the random-embedding control through one concrete
          integer. The four steps are{" "}
          <strong>rendering → tokenization → embedding lookup → mean-pooling</strong>.
          Each one turns the input into something the next step can
          act on. The control swaps step 3 only.
        </p>

        <p className="mt-3">
          <strong>Step 1: rendering.</strong>{" "}
          <span className="font-mono">format_number(23, "babylonian")</span>{" "}
          returns "𒌋𒌋𒁹𒁹𒁹": two ten-wedges followed by three
          one-wedges. The structure is <em>additive</em>: the number
          of 𒌋 equals the tens digit, the number of 𒁹 equals the
          ones digit. This is the renderer's job — it turns the
          integer into a string whose shape encodes the arithmetic
          of the integer.
        </p>

        <p className="mt-3">
          <strong>Step 2: tokenization.</strong> Prepend a space and
          feed " 𒌋𒌋𒁹𒁹𒁹" to Pythia's byte-level BPE tokenizer.
          Cuneiform wasn't in Pythia's training corpus, so each
          codepoint falls back to its UTF-8 byte representation
          (4 bytes per cuneiform codepoint, often each byte becoming
          its own sub-token). Crucially, <strong>every 𒌋 expands to
          the same fixed byte-token sequence</strong>, and likewise
          for 𒁹. If we write <span className="font-mono">T_𒌋</span>{" "}
          for the sequence of byte-tokens making up one 𒌋, the
          tokenization of 23 is:
        </p>

        <MM>{String.raw`[\,\text{space},\; T_{\text{𒌋}},\; T_{\text{𒌋}},\; T_{\text{𒁹}},\; T_{\text{𒁹}},\; T_{\text{𒁹}}\,]`}</MM>

        <p>
          A constant prefix plus 2 copies of{" "}
          <M>{String.raw`T_{\text{𒌋}}`}</M> plus 3 copies of{" "}
          <M>{String.raw`T_{\text{𒁹}}`}</M>. The <em>count</em>{" "}
          structure of step 1 is preserved by the tokenizer.
        </p>

        <p className="mt-3">
          <strong>Step 3: embedding lookup.</strong> Each token ID
          indexes into the embedding matrix{" "}
          <M>{String.raw`E \in \mathbb{R}^{V \times d}`}</M> (for
          Pythia, <M>{String.raw`V \approx 50{,}000`}</M> and{" "}
          <M>d = 4096</M>). We get one vector per sub-token. If we
          write <M>{String.raw`e_{\text{𒌋}}`}</M> for the contribution
          of one <M>{String.raw`T_{\text{𒌋}}`}</M> (the sum of the
          four byte-token rows), the sequence of vectors is:
        </p>

        <MM>{String.raw`[\,e_{\text{space}},\; e_{\text{𒌋}},\; e_{\text{𒌋}},\; e_{\text{𒁹}},\; e_{\text{𒁹}},\; e_{\text{𒁹}}\,]`}</MM>

        <p className="mt-3">
          <strong>Step 4: mean-pooling.</strong> Average the rows:
        </p>

        <MM>{String.raw`h(23) \;\approx\; \frac{e_{\text{space}} \;+\; 2\, e_{\text{𒌋}} \;+\; 3\, e_{\text{𒁹}}}{6}`}</MM>

        <p>
          This is the L=0 representation of the integer 23. It's a{" "}
          <strong>linear combination of three fixed vectors</strong>{" "}
          with coefficients that come from the renderer's count
          structure: 1 for the space, 2 (= the tens digit of 23) for{" "}
          <M>{String.raw`e_{\text{𒌋}}`}</M>, 3 (= the ones digit) for{" "}
          <M>{String.raw`e_{\text{𒁹}}`}</M>.
        </p>

        <p className="mt-3">
          Generalising to all <M>n &lt; 60</M>:
        </p>

        <MM>{String.raw`h(n) \;\approx\; \alpha(n)\, e_{\text{𒌋}} \;+\; \beta(n)\, e_{\text{𒁹}} \;+\; \text{constant}`}</MM>

        <p>
          where <M>{String.raw`\alpha(n) = \lfloor n/10 \rfloor`}</M> and{" "}
          <M>{String.raw`\beta(n) = n \bmod 10`}</M>. Both are
          smooth/periodic functions of n.
        </p>

        <p className="mt-4">
          <strong>What the random-embedding control changes.</strong>{" "}
          Steps 1, 2, 4 are unchanged. Only step 3 changes:{" "}
          <M>{String.raw`e_{\text{𒌋}}`}</M> and{" "}
          <M>{String.raw`e_{\text{𒁹}}`}</M> are replaced by fresh
          draws from <M>{String.raw`\mathcal{N}(0, 1/\sqrt{d})`}</M>. So:
        </p>

        <MM>{String.raw`\tilde{h}(n) \;\approx\; \alpha(n)\, \tilde{e}_{\text{𒌋}} \;+\; \beta(n)\, \tilde{e}_{\text{𒁹}} \;+\; \text{constant}`}</MM>

        <p>
          The vectors <M>{String.raw`\tilde{e}_{\text{𒌋}}`}</M> and{" "}
          <M>{String.raw`\tilde{e}_{\text{𒁹}}`}</M> now point in
          random directions in <M>{String.raw`\mathbb{R}^{4096}`}</M>,
          but they're still <em>fixed</em> vectors, and the
          coefficients <M>{String.raw`\alpha(n)`}</M> and{" "}
          <M>{String.raw`\beta(n)`}</M> are the same smooth functions
          of n. The trig basis fits the coefficient structure
          regardless of which directions{" "}
          <M>{String.raw`\tilde{e}_{\text{𒌋}}`}</M> and{" "}
          <M>{String.raw`\tilde{e}_{\text{𒁹}}`}</M> point:{" "}
          <M>{String.raw`\beta(n) = n \bmod 10`}</M> is fit by the
          T=10 cos/sin columns,{" "}
          <M>{String.raw`\alpha(n) = \lfloor n/10 \rfloor`}</M> is
          approximately linear in n for <M>n &lt; 60</M> so the linear
          column fits it.
        </p>

        <p className="mt-3">
          The <em>direction</em> of the fitted helix in residual
          space rotates with each random seed; the <em>quality</em>{" "}
          of the fit (R²) doesn't change.
        </p>

        <p className="mt-4">
          <strong>Why Latin is different.</strong> Pythia tokenizes
          integers in [0, 557] as <em>single tokens</em>, so for
          n=23 on Latin the chain collapses:
        </p>

        <MM>{String.raw`\texttt{" 23"} \;\longrightarrow\; [\, T_{23}\, ] \;\longrightarrow\; [\, e_{23}\, ]`}</MM>

        <p>
          Mean-pooling over one token is the identity. So{" "}
          <M>{String.raw`h(23) = e_{23}`}</M>, and the helix score is
          asking whether the 100 <em>learned embedding vectors</em>{" "}
          <M>{String.raw`\{e_0, e_1, \ldots, e_{99}\}`}</M> themselves
          lie on a helix. With learned Pythia weights they
          do — Pythia learned that "23" is the embedding-space
          neighbour of "22" and "24". Replace E with a random matrix
          and the 100 vectors are independent random points; the
          helix R² collapses to chance.
        </p>

        <p className="mt-4">
          So the <strong>same control</strong> tells very different
          stories on different cells:
        </p>

        <ul className="mt-2 list-disc pl-6 space-y-1">
          <li>
            <strong>Latin on Pythia</strong>: large positive Δ ≈ 0.10.
            The L=0 helix is carried by the learned embeddings
            themselves; no rendering or pooling chain to fall back on
            (one token, one vector).
          </li>
          <li>
            <strong>Babylonian on Pythia</strong>: Δ ≈ 0. The L=0
            helix is generated by the rendering+tokenization+pooling
            chain operating on counts, and the embedding values
            don't matter.
          </li>
        </ul>

        <p className="mt-4">
          This is what we mean by "geometry from rendering alone."
          It isn't mystical — it's the deterministic consequence of
          (i) an additive renderer that turns n into symbol counts,
          (ii) a tokenizer that preserves those counts in the token
          sequence, and (iii) mean-pooling that turns counts into
          convex combinations of fixed vectors. The trig basis fits
          the convex-combination weights as smooth functions of n —
          which is exactly what we asked it to do.
        </p>

        <p className="mt-3">
          The cuneiform-ness is incidental. The same effect would
          appear for <em>any</em> script with (additive symbolic
          renderer) + (count-preserving tokenizer) + (mean pooling).
          Cuneiform is just the cleanest case because the model has
          near-zero pre-training exposure to it, so we can't even
          tell ourselves a "the embeddings carry learned semantics"
          story.
        </p>
      </details>

      <p>
        This is the cleanest single result in the paper, and it makes a
        sharper general point: <strong>some apparent neural geometry
        can arise from rendering statistics alone</strong>. When a
        residual stream "encodes" a quantity, the encoding can come from
        arithmetic done in the renderer before the model has read the
        tokens. Provenance analysis needs to extend below L=0 — to the
        rendering pipeline itself.
      </p>

      <p>
        One twist (foreshadowing Finding 5): even though Babylonian's
        L=0 helix is mechanical, the representation-alignment check
        shows that Pythia, Llama, and OLMo end up with a substantially
        different integer-to-integer geometry by the peak layer
        (CKA = 0.25–0.53). The peak Babylonian helix is not the
        mechanical L=0 helix passed through — it's a depth-built
        object, just one whose <em>score</em> happens to match the
        mechanical starting point. What that rebuilt helix is doing
        computationally is a separate, causal question.
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
          <tr><td>GPT-J-6B</td><td className="numeric text-right text-accent">+0.12</td><td className="numeric text-right text-ink-mute">−0.002</td></tr>
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
