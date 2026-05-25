/**
 * Names the three failure modes of the standard "fit a helix R²"
 * protocol up front, introduces ρ and CKA as the two main diagnostics,
 * and includes a collapsible methods box at the bottom for the precise
 * pipeline. The cell-by-cell visualisations live in Part 3.
 */
export function TheDiagnostic() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">A diagnostic kit for helix R²</h2>

      <p>
        The standard procedure — take 100 integers, fit a 9-feature
        trigonometric basis to the model's internal state, read at one
        layer — was designed for Pythia and Llama on Latin digits. Apply
        it anywhere else and three problems can make it answer the wrong
        question.{" "}
        <strong>Two of them hide real structure that's actually there;
        one credits the transformer with structure it didn't build.</strong>
      </p>

      <ol className="mt-6 space-y-4 list-decimal pl-6">
        <li>
          <strong>The basis can't see the right period.</strong> The
          standard fit is to a fixed set of periods (2, 5, 10, 100). If
          the model is encoding something at a different period, the fit
          looks weak even when the structure is strong.{" "}
          <em>The fix:</em> cross-check against a basis-free FFT and add
          any missing periods. (Babylonian's helix score jumps 8–16
          points once you include T = 60.)
        </li>
        <li>
          <strong>The input range is too short.</strong> To detect a
          period <span className="font-mono">T</span> by FFT you need
          several wraps of <span className="font-mono">T</span> in the
          input. The default 0..99 range is fine for T = 2, 5, 10 (50,
          20, 10 wraps), barely workable for T = 100 (one wrap), and{" "}
          <em>not</em> workable at all for T = 60 (one wrap, misaligned).{" "}
          <em>The fix:</em> widen the input range until any candidate
          period wraps many times.
        </li>
        <li>
          <strong>The input pipeline supplies the geometry — the
          transformer just passes it through.</strong> A great helix fit
          at the best layer can be the entire output of how the model
          renders the number to text, breaks it into tokens, looks each
          token up in the embedding table, and averages the results. The
          transformer might have done nothing to construct it. This is
          the representation-level analogue of the tokenization effects
          studied by{" "}
          <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2402.14903">
            Singh &amp; Strouse
          </a>: tokenization is not a neutral preprocessing detail, but
          an inductive bias in the numerical reasoning pipeline. You
          can't tell which case you're in without comparing the best
          layer to the very first layer (before any computation).{" "}
          <em>The fix:</em> alongside the peak score, report the ratio
        </li>
      </ol>

      <p className="mt-3 ml-12 font-mono text-sm text-ink/80">
        ρ = R²(L=0) / max<sub>L</sub> R²(L).
      </p>

      <p className="mt-3 ml-12">
        ρ ≈ 1 means depth didn't improve the helix score — the structure
        was already there before any transformer block ran. This is a
        provenance check, not a causal proof: it doesn't show the
        transformer ignores the geometry, only that the geometry didn't
        need depth to exist. (To distinguish "depth left the helix
        alone" from "depth produced a similar-quality helix in a
        different direction" needs a second check — see Finding 5.)
      </p>

      <p className="mt-6">
        In the original Kantamneni &amp; Tegmark setup (Pythia, Latin,
        paper basis, n=100), all three modes are silent: the basis is
        right, the window wraps, and depth genuinely builds the helix.
        The rest of this post is what happens when each assumption
        breaks.
      </p>

      <p className="mt-4">
        The 8-model × 12-script matrix below is the test bed. It exists to
        surface the three modes, not to publish "the helix generalises to
        twelve numeral systems!" — that would have been the headline from
        a naive read, and ρ is what stops you writing it.
      </p>

      <h3 className="section-subheading">A second check: representation alignment (CKA)</h3>

      <p>
        ρ has a known blind spot: a high ρ can mean either the
        transformer left the geometry alone (true pass-through){" "}
        <em>or</em> the transformer produced a similar-quality helix
        with a substantially different integer-to-integer pattern (a
        rebuild). To separate those, I add <strong>CKA</strong>{" "}
        (Centered Kernel Alignment) — a standard similarity metric for
        neural representations that scores how much the pairwise
        pattern of distances between integers at one layer matches the
        pattern at another. CKA near 1 means the geometries are
        essentially the same; near 0 means unrelated. Combining ρ and
        CKA gives a four-way classification of every cell — see the ρ
        heatmap, the classification grid, and the CKA scatter in Part 3.
      </p>

      <details className="mt-8 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-3">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Methods box
        </summary>

        <p>
          <strong>The pipeline.</strong> For each combination of (model,
          numeral system, integer <span className="font-mono">a</span>):
        </p>
        <ol className="mt-1 list-decimal pl-6 space-y-1">
          <li>
            Render <span className="font-mono">a</span> as a string in the
            chosen numeral system (see "The twelve numeral systems"
            above).
          </li>
          <li>
            Break that string into tokens using the model's own
            tokenizer — the same rules the model uses for any text. No
            prompt scaffolding, just the bare number.
          </li>
          <li>
            Run the model on those tokens and capture the internal state
            after every layer.
          </li>
          <li>
            <strong>Pool</strong> the activations across the numeral's
            tokens by taking the mean. (The alternative — reading only
            the last sub-token — bakes a per-digit cycle into the data
            for multi-digit scripts, producing fake helix peaks; mean
            is a more symmetric choice across scripts.{" "}
            <em>That said, mean-pooling is itself part of the provenance
            story: for additive renderings like Babylonian and binary,
            it turns symbol counts into linear arithmetic on token
            embeddings — see Findings 3 and the binary/hex section.</em>)
          </li>
          <li>
            Collect these per-integer vectors into a matrix per layer —
            100 rows by default (one per integer 0..99), 600 for the
            Babylonian wider-window run, 1024 for binary/hex.
          </li>
        </ol>

        <p className="mt-3">
          <strong>How layers are numbered.</strong> I refer to layer 0
          as the model's first internal state, before any transformer
          block has run — just the result of looking up each token in
          the embedding table (plus positional info, for models that
          add it there). Layer <span className="font-mono">k</span> for{" "}
          <span className="font-mono">k ≥ 1</span> is the internal state
          after block <span className="font-mono">k</span> has run. A
          model with N transformer blocks therefore has N+1 layer
          readouts available.
        </p>

        <p className="mt-3">
          <strong>The helix fit.</strong> The basis I fit is one straight
          ("number-line") direction plus four cosine/sine pairs at
          periods 2, 5, 10, and 100 — nine features in total, one per
          integer. I fit it by ordinary least squares (the standard way
          to find the best linear approximation) and report the variance
          explained as helix R². As a comparison ceiling, I also compute
          the best possible 9-dimensional approximation of the same
          data (a textbook result called the Eckart–Young theorem) and
          call it the <strong>9-d PCA upper bound</strong>. The{" "}
          <strong>helix / PCA quality ratio</strong> is helix R² divided
          by that ceiling; values near 1 mean the trig basis isn't just{" "}
          a good 9-D fit — it <em>is</em> the 9-D structure of the
          activations. This ratio is the "quality" axis in the
          classification grid below.
        </p>

        <p className="mt-3">
          <strong>Native bases for non-decimal scripts.</strong> The
          paper-default basis [2, 5, 10, 100] catches only base-10
          structure. For scripts whose natural period doesn't fit, I
          additionally run a basis tuned to the script:
        </p>
        <ul className="mt-1 list-disc pl-6 space-y-1 text-[0.95rem]">
          <li>
            <strong>Babylonian</strong> (base 60): basis [2, 5, 10, 60,
            100], n = 600 (ten wraps of T=60).
          </li>
          <li>
            <strong>Binary</strong> (base 2): basis [2, 4, 8, 16, 32, 64],
            n = 1024 (≥ 16 wraps of any period below 64).
          </li>
          <li>
            <strong>Hexadecimal</strong> (base 16): basis [16, 32, 64,
            256], n = 1024.
          </li>
        </ul>

        <p className="mt-3">
          <strong>Peak layer.</strong> The single layer (between 0 and
          N) at which helix R² is highest for that cell, found by
          sweeping the basis fit across every layer of the model. ρ
          uses this as the denominator; CKA uses it as one of the two
          endpoints to compare.
        </p>

        <p className="mt-3">
          <strong>FFT and PCA as auxiliary tools.</strong> Two further
          views appear in the body. The FFT panel shows the magnitude
          of each periodic component in the residual stream — used as
          a basis-free sanity check (does the basis include the periods
          the model actually uses?). I also auto-label the top-5 peaks
          by prominence to surface unexpected periodicities. The 2-D
          PCA scatter is the first two principal components of the
          residual-stream matrix at the peak layer — useful for
          identifying non-helical structure (e.g., the staircase shape
          of Roman, the tight letter-clusters of Greek alphabetic).
        </p>

        <p className="mt-3">
          <strong>The provenance ratio ρ.</strong>
        </p>

        <div className="my-3 flex items-center justify-center gap-3 text-[1.05rem] font-serif">
          <span className="italic">ρ</span>
          <span>=</span>
          <div className="inline-flex flex-col items-center leading-tight">
            <span className="px-3 pb-0.5">
              <span className="italic">R</span><sup>2</sup>
              <sub className="text-[0.75em]">helix</sub>
              <span>(</span><span className="italic">L</span> = 0<span>)</span>
            </span>
            <span className="border-t border-ink/50 px-3 pt-1 w-full text-center">
              max<sub className="text-[0.75em] italic">L</sub>{" "}
              <span className="italic">R</span><sup>2</sup>
              <sub className="text-[0.75em]">helix</sub>
              <span>(</span><span className="italic">L</span><span>)</span>
            </span>
          </div>
        </div>

        <p>
          ρ near 1 means depth didn't improve the score (so the geometry
          was already there); ρ small means depth substantially built
          it. For the cell-by-cell classification I use mutually
          exclusive joint criteria, ordered most-restrictive first:
        </p>

        <ul className="my-3 list-disc pl-6 space-y-1 text-[0.95rem]">
          <li><strong>depth-built</strong>: ρ ≤ 0.55 and helix/PCA ≥ 0.70</li>
          <li><strong>depth-amplified</strong>: 0.55 &lt; ρ ≤ 0.70 and helix/PCA ≥ 0.60</li>
          <li><strong>inherited</strong>: ρ ≥ 0.80 and helix/PCA ≥ 0.55</li>
          <li><strong>weak</strong>: helix/PCA &lt; 0.55 (basis fits little, regardless of ρ)</li>
          <li><strong>ambiguous</strong>: everything else (e.g. 0.70 &lt; ρ &lt; 0.80)</li>
        </ul>

        <p>
          These thresholds apply to the <em>paper-default</em> protocol
          (n=100, basis [2,5,10,100]). For scripts where that protocol
          is known to misfit (binary, hex, Babylonian), the meaningful ρ
          comes from the native-basis run; the classification grid above
          excludes them for that reason.
        </p>

        <p className="mt-3">
          <strong>How I handle zero.</strong> Positional base-10 scripts
          (Latin, Arabic-Indic, Persian, Devanagari, Thai, CJK) all have
          a native zero. Scripts that historically don't, I use a
          placeholder: <span className="font-mono">"nulla"</span> for
          Roman, <span className="font-mono">"Ø"</span> for Greek, the
          late-period cuneiform zero glyph for Babylonian, and the
          modern Hebrew word for zero (<span className="font-mono">אפס</span>)
          for Hebrew. Dropping <span className="font-mono">a = 0</span>{" "}
          from the analysis doesn't change any of the headline numbers.
        </p>

        <p className="mt-3">
          <strong>Random-embedding control.</strong> To test whether a
          layer-0 helix reflects learned numeric structure or just the
          mechanical effect of rendering + tokenization + averaging, I
          replace the model's learned embedding table with a fresh
          random one of the same shape (vocab × hidden), drawn from
          <span className="font-mono"> N(0, 1/√d)</span> where{" "}
          <span className="font-mono">d</span> is the model's hidden
          dimension. I then re-run only the tokenize-and-pool steps.
          If the random version produces the same helix score as the
          real one, the structure was never in the learned weights to
          begin with. Run on all 12 scripts × 8 models, and on the
          native bases for binary/hex/Babylonian; see Finding 3 for the
          comparison. Implemented in{" "}
          <span className="font-mono">embed_control.py</span>.
        </p>

        <p className="mt-3">
          <strong>Subspace alignment (CKA).</strong> CKA (Centered
          Kernel Alignment) is a standard similarity metric for neural
          representations. It scores how much the pairwise pattern of
          distances between integers at one layer matches the pattern
          at another — 1 = identical structure, 0 = unrelated, and
          invariant to rotation and rescaling. I compute it between
          layer 0 and the best layer, both for the full activations
          and just for the helix-shaped slice of them. Together with
          ρ this distinguishes "depth left the geometry alone" from
          "depth produced a similar-quality helix in a different
          direction." Run on all 8 models × 12 scripts; see Finding 5.
          Implemented in{" "}
          <span className="font-mono">subspace_align.py</span>.
        </p>

        <p className="mt-3">
          <strong>What's <em>not</em> here.</strong> No causal
          intervention experiments — I don't manipulate the helix and
          check whether downstream computation changes (that's the
          Goodfire complement). No held-out cross-validation of the
          basis fit. Only one prompt template per script. And the
          comparison to a random-embedding control covers the mechanical
          /-learned axis at layer 0 only; later layers haven't been
          tested the same way.
        </p>
      </details>
    </section>
  );
}
