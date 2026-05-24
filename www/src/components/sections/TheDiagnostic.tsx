/**
 * Slots between WhatTheHelixIs and Finding1Layers. Names the three
 * failure modes of the standard "fit a helix R²" protocol up front,
 * and frames the matrix sections that follow as applications of the
 * diagnostic, not standalone findings.
 */
export function TheDiagnostic() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">A diagnostic kit for helix R²</h2>

      <p>
        The standard protocol — fit a 9-feature trig basis (one linear
        column, four (cos, sin) pairs at <span className="font-mono">T ∈ {"{"}2, 5, 10, 100{"}"}</span>) to a
        residual stream of 100 integers, read at one layer — was designed
        for Pythia and Llama on Latin. Apply it anywhere else and three
        failure modes systematically inflate the reported result. They
        share a direction: each one quietly flatters the protocol.
      </p>

      <ol className="mt-6 space-y-4 list-decimal pl-6">
        <li>
          <strong>The basis has bandwidth.</strong> Helix R² is a regression
          fit to a fixed set of periods. If the model encodes something at
          a period the basis doesn't include, the metric reads low even
          when the structure is strong. <em>The fix:</em> cross-check
          against the basis-free FFT and add missing periods. (Babylonian
          gains 8–13 percentage points just from adding T=60.)
        </li>
        <li>
          <strong>The window must wrap.</strong> To detect period{" "}
          <span className="font-mono">T</span> via FFT you need several
          wraps of <span className="font-mono">T</span> in the input range.
          The default 0..99 is fine for T = 2, 5, 10 (50, 20, 10 wraps),
          barely workable for T = 100 (one wrap), and <em>not</em>{" "}
          workable for T = 60 (one wrap, phase-misaligned).{" "}
          <em>The fix:</em> widen the window to many wraps of any
          candidate period.
        </li>
        <li>
          <strong>The tokeniser inherits.</strong> A high helix R² at the
          peak layer can be entirely inherited from the L=0 embedding
          mean, and reads <em>identically</em> to one the transformer
          built. <em>The fix:</em> alongside the peak, report the L=0 /
          peak ratio{" "}
          <span className="font-mono">ρ = R²(L=0) / max_L R²(L)</span>.{" "}
          ρ ≈ 1 means the structure was already there before any block ran.
        </li>
      </ol>

      <p className="mt-6">
        The first two are about <em>under</em>-detection — measurement
        choices that hide real structure. The third is about{" "}
        <em>over</em>-detection — choices that make structure look
        depth-built when it's just preserved from the embeddings. In the
        Kantamneni &amp; Tegmark setup (Pythia, Latin, paper basis,
        n=100), all three are silent because the basis is right, the
        window wraps, and depth genuinely builds the helix. The rest of
        this post is what happens when each assumption breaks.
      </p>

      <p className="mt-4">
        The 7-model × 8-script matrix below is the test bed. It exists to
        surface the three modes, not to publish "the helix generalises to
        eight numeral systems!" — that would have been the headline from a
        naive read, and ρ is what stops you writing it.
      </p>
    </section>
  );
}
