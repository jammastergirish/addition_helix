import { RhoCkaScatter } from "../charts/RhoCkaScatter";

/**
 * Subspace alignment (linear CKA) between L=0 and peak-layer fitted
 * helix subspaces. Goes after the ρ heatmap and before the Conclusion.
 *
 * Computed across all 8 models × 12 scripts (subspace_align.py).
 */
export function Finding5CKA() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">
        Representation alignment: preserved, amplified, or rebuilt?
      </h2>

      <p>
        ρ is a single number — it just compares the helix fit at the
        first layer to the helix fit at the best layer. Two cells with
        the same ρ can be mechanistically different: depth might leave
        the representation alone, refine it, or replace it with a
        differently-shaped one of similar quality. To tell these apart,
        I also measure how similar the two representations are to each
        other, using a standard metric called{" "}
        <strong>CKA</strong> (linear Centered Kernel Alignment).
      </p>

      <p>
        CKA is a number between 0 and 1 that captures{" "}
        <em>representation similarity</em> across the 100 example
        integers: <strong>1 means the pattern of distances between
        integers at layer 0 matches the pattern at the best layer</strong>,
        <strong> 0 means the two patterns are unrelated</strong>. CKA is
        invariant to rotation and rescaling — it doesn't measure overlap
        of any particular hidden direction, only whether the same
        integer-to-integer geometry is present. The most informative
        version measures CKA on just the helix-shaped slice of the
        activations (project both onto the fitted basis first) — that
        filters out everything else the model is doing and isolates
        whether the <em>helix</em> looks the same. Combining ρ and CKA
        gives four regimes:
      </p>

      <figure className="my-8">
        <div className="rounded-lg border border-ink/10 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <RhoCkaScatter />
        </div>
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          Each point is one (model, script) cell. The two axes are
          provenance (ρ — does depth raise the helix score?) and
          representation alignment (CKA — is the same
          integer-to-integer geometry present at L=0 and at the peak
          layer?). Hover for cell details.
        </figcaption>
      </figure>

      <h3 className="section-subheading">Three patterns that ρ alone would conflate</h3>

      <p>
        Each point in the scatter above is one (model, script) cell. The
        x-axis is provenance (does depth improve the score?). The y-axis
        is similarity (does the geometry look the same before and after
        depth?). Reading the four corners clockwise from the top-right:
      </p>

      <ul className="mt-4 space-y-3 list-disc pl-6">
        <li>
          <strong>Pass-through</strong> (high ρ, high CKA — the orange
          quadrant). Depth doesn't improve the score AND the
          integer-to-integer geometry looks the same. Examples:
          Qwen-32B/Latin (ρ = 0.89, CKA = 0.91), OLMo/Latin (0.63, 0.91),
          Pythia × Persian/Devanagari/Chinese (CKA ≥ 0.98). The
          first-layer helix passes through the transformer essentially
          untouched — the clearest case where the input pipeline is
          doing all the work.
        </li>
        <li>
          <strong>Rebuilt with a different geometry</strong> (high ρ, low
          CKA — the red quadrant, and the most surprising finding). The
          score is unchanged, but the integer-to-integer distance pattern
          isn't — depth replaces the first-layer helix with a
          similar-quality one whose row-similarity geometry is
          substantially different. The whole Gemma family lives here
          (CKA 0.21–0.65 across most scripts).{" "}
          <strong>Babylonian on Pythia, Llama, and OLMo</strong> sits
          here too (ρ ≈ 0.86, CKA = 0.25–0.53) — depth produces a
          different helix on top of the mechanical first-layer one.{" "}
          <strong>Qwen on Devanagari</strong> is the same (ρ ≈ 0.76, CKA
          ≈ 0.24–0.30). ρ alone would lump all of these with the orange
          (pass-through) cells; CKA shows depth is doing real work, just
          not improving the score. (Note: CKA is invariant to rotation,
          so "different geometry" here means the row-similarity pattern
          across integers differs — not necessarily that the hidden
          directions are orthogonal.)
        </li>
        <li>
          <strong>Depth refines what's there</strong> (low-to-moderate ρ,
          high CKA — the green quadrant). Pythia/Latin (0.34, 0.79) and
          Llama/Latin (0.39, 0.81) — the paper's clean cells. Also Roman
          on every non-Gemma model (CKA = 0.77–0.90): even when the
          first-layer helix is weak, depth preserves and amplifies its
          integer-to-integer geometry rather than rebuilding from scratch.
          Gemma is the lone exception (Roman CKA 0.30, 0.57) — Gemma
          rebuilds Roman like it rebuilds everything else.
        </li>
        <li>
          <strong>Depth builds something new</strong> (low ρ, low CKA —
          the grey quadrant). Empirically the rarest case in this
          matrix. Gemma's Devanagari/Persian/Roman/Greek cells edge into
          it (ρ ≈ 0.44–0.63, CKA ≈ 0.29–0.57) — depth is doing the most
          representational work on Gemma's non-Latin scripts, building a
          helix with a substantially different integer-to-integer
          geometry than the input pipeline supplied.
        </li>
      </ul>


      <h3 className="section-subheading">Two things the matrix forces a rewrite of</h3>

      <p>
        <strong>1. Gemma's "high ρ" isn't really tokenizer inheritance.</strong>{" "}
        The simpler reading — "Gemma/Latin ρ ≈ 0.85, so mostly tokenizer"
        — was wrong. CKA on the same cell is only 0.33. Depth produces a
        helix of similar quality, but with a substantially different
        integer-to-integer geometry. Across nearly every script, Gemma
        rebuilds. ρ alone made it look like Gemma was doing nothing;
        it's actually doing a lot, just in a way that doesn't change
        the score.
      </p>

      <p>
        <strong>2. Babylonian's first-layer helix is mechanical, but
        the peak isn't.</strong> The random-embedding control in
        Finding 3 showed Babylonian's layer-0 helix is just rendering
        + tokenization + pooling. But ρ + CKA together show that Pythia,
        Llama, and OLMo end up with a substantially different
        integer-to-integer geometry by the peak layer (CKA = 0.25–0.53).
        The peak Babylonian helix is depth-built — built on top of a
        mechanical starting point. Whether the rebuilt helix is actually
        used for anything downstream is a separate, causal question.
      </p>

      <p className="mt-6 text-sm text-ink-mute">
        Cells with peak layer = 0 are trivially aligned (CKA(L=0, L=0) =
        1) and shown faded in the scatter. They aren't informative for
        the alignment question — they're effectively the same point as
        ρ = 1.
      </p>
    </section>
  );
}
