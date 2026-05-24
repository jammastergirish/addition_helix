import { RhoCkaScatter } from "../charts/RhoCkaScatter";

/**
 * Subspace alignment (linear CKA) between L=0 and peak-layer fitted
 * helix subspaces. Goes after the ρ heatmap and before the Conclusion.
 *
 * Computed across all 8 models × 8 scripts (subspace_align.py).
 */
export function Finding5CKA() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">
        Subspace alignment: preserved, amplified, or rebuilt?
      </h2>

      <p>
        ρ is a scalar — it compares helix R² at L=0 to helix R² at peak.
        Two cells with the same ρ can be mechanistically different: depth
        might preserve the L=0 subspace, amplify it, or destroy it and
        rebuild a helix of similar quality in a new subspace. To
        distinguish these, I additionally measure <strong>linear CKA</strong>{" "}
        (Centered Kernel Alignment) between the L=0 and peak
        representations, on the same 100 integers per cell.
      </p>

      <p>
        CKA is in [0, 1]; 1 means the two representations carry the same
        row-similarity structure (i.e., the pattern of distances between
        integers is preserved), 0 means unrelated. The most informative
        version is <strong>CKA on the fitted helix subspace only</strong>{" "}
        (project both into <span className="font-mono">B · W</span>{" "}
        before computing) — this isolates the helix-relevant directions
        from everything else the residual stream carries. The two
        diagnostics together give four mechanistic regimes:
      </p>

      <figure className="my-8">
        <div className="rounded-lg border border-ink/10 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <RhoCkaScatter />
        </div>
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          Each point is one (model, script) cell. The two axes are
          provenance (ρ — does depth raise the helix score?) and subspace
          alignment (CKA — is the peak helix in the same subspace as L=0?).
          Hover for cell details.
        </figcaption>
      </figure>

      <h3 className="section-subheading">Three patterns that ρ alone would conflate</h3>

      <p>
        Reading the scatter clockwise from the top-right:
      </p>

      <ul className="mt-4 space-y-3 list-disc pl-6">
        <li>
          <strong>Pass-through (high ρ, high CKA)</strong> — the orange
          quadrant. Qwen-32B/Latin (ρ=0.89, CKA=0.91), OLMo/Latin
          (0.63, 0.91), Pythia × Persian/Devanagari/Chinese
          (CKA ≥ 0.98), Qwen/Chinese, Qwen/Greek. The L=0 helix passes
          through the transformer essentially untouched. These are the
          cells most clearly attributable to the tokenizer/embedding
          front end.
        </li>
        <li>
          <strong>Rebuilt at a different subspace (high ρ, low CKA)</strong>{" "}
          — the red quadrant, and the most surprising finding. The whole
          Gemma family lives here (CKA 0.19–0.65 across most scripts).{" "}
          <strong>Babylonian on Pythia, Llama, and OLMo</strong> is here
          too (ρ ≈ 0.86, CKA = 0.25–0.53) — the mechanical L=0 helix gets
          rebuilt at a substantially different subspace by depth.{" "}
          <strong>Qwen on Devanagari</strong> sits here at ρ ≈ 0.76, CKA
          ≈ 0.24–0.30. ρ alone would lump all of these with the orange
          (pass-through) cells; CKA shows depth is doing real work, just
          not improving the helix R² score.
        </li>
        <li>
          <strong>Depth amplifies (low-moderate ρ, high CKA)</strong> —
          the green quadrant. Pythia/Latin (0.34, 0.79) and Llama/Latin
          (0.39, 0.81) — the paper's clean cells. Also Roman on every
          model (CKA = 0.82–0.90), which is interesting: even when ρ
          climbs, the L=0 and peak Roman helices stay in roughly the
          same subspace. The additive Roman renderer hands the
          transformer a structure that gets refined in place rather than
          rotated away from.
        </li>
        <li>
          <strong>Depth builds a new helix (low ρ, low CKA)</strong> — the
          grey quadrant. Empirically the rarest case in this matrix.
          Gemma's Devanagari/Persian/Roman/Greek cells edge into it
          (ρ ≈ 0.44–0.61, CKA ≈ 0.30–0.45). When depth is doing the
          most representational work on Gemma's non-Latin scripts, the
          subspace at the peak is genuinely new.
        </li>
      </ul>

      <h3 className="section-subheading">Two things the matrix forces a rewrite of</h3>

      <p>
        <strong>1. Gemma's "high ρ" is not tokenizer inheritance.</strong>{" "}
        The previous reading — "Gemma/Latin ρ ≈ 0.85, mostly tokeniser" —
        was wrong. CKA = 0.33 on the same cell. Depth produces a helix
        of similar quality, but in a substantially different subspace.
        Across nearly every script, Gemma rebuilds. This is a Gemma
        family characteristic that ρ flattens.
      </p>

      <p>
        <strong>2. Babylonian's L=0 is mechanical, but the peak isn't.</strong>{" "}
        The random-embedding control in Finding 3 showed the L=0
        Babylonian helix is rendering-statistics-driven. But ρ + CKA
        together show: Pythia, Llama, and OLMo then <em>rebuild</em>{" "}
        that helix at a different subspace (CKA = 0.25–0.53). The peak
        Babylonian helix is depth-built — just on top of a mechanical
        starting point. Whether the rebuilt helix is doing arithmetic or
        something else is the next causal question.
      </p>

      <p className="mt-6 text-sm text-ink-mute">
        Cells with peak layer = 0 are trivially aligned (CKA(L=0, L=0)
        = 1) and shown faded in the scatter. These are not informative
        for the alignment question — they're effectively the same point
        as ρ = 1.
      </p>
    </section>
  );
}
