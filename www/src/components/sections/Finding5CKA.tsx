import { RhoCkaScatter } from "../charts/RhoCkaScatter";
import { M, MM } from "../Math";

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

      <details className="my-6 rounded-md border border-ink/10 bg-paper-warm/40 px-4 py-3 text-[0.97rem] leading-relaxed text-ink-soft [&[open]>summary]:mb-2">
        <summary className="cursor-pointer select-none font-sans text-sm font-medium text-ink/80 hover:text-accent">
          ▸ Deep dive: <em>same ρ, different CKA — a worked comparison</em>
        </summary>

        <p>
          Two cells, both with high ρ — meaning the helix score at the
          peak layer is mostly already at L=0. By ρ alone they look the
          same. CKA reveals two very different mechanisms.
        </p>

        <p className="mt-3">
          <strong>Setup.</strong> For each cell:
        </p>
        <ol className="mt-2 list-decimal pl-6 space-y-1">
          <li>Run the layer sweep, identify the peak layer.</li>
          <li>
            Collect H at L=0 and at the peak layer — two matrices of
            shape (100, 4096), one row per integer.
          </li>
          <li>
            Fit the helix basis on each →{" "}
            <M>{String.raw`W^{(L=0)}`}</M> and{" "}
            <M>{String.raw`W^{(\text{peak})}`}</M>.
          </li>
          <li>
            Form the helix-projected representations{" "}
            <M>{String.raw`X = B \cdot W^{(L=0)}`}</M> and{" "}
            <M>{String.raw`Y = B \cdot W^{(\text{peak})}`}</M> — the
            best 9-dimensional helix approximations of each layer,
            expanded back into the full 4096-d hidden space.
          </li>
          <li>Compute <M>{String.raw`\mathrm{CKA}(X, Y)`}</M>.</li>
        </ol>

        <p className="mt-3">
          <strong>Cell A: Qwen2.5-32B/Latin.</strong> ρ ≈ 0.89. The
          helix at L=0 is already near the peak score (Qwen splits
          Latin numbers per-digit, so rendering + tokenization +
          mean-pooling supplies a fittable manifold before any block
          has run). CKA ≈ 0.91.
        </p>

        <p className="mt-2 ml-4">
          Reading: both the score <em>and</em> the integer-to-integer
          relationships are preserved through depth.{" "}
          <strong>Pass-through</strong>. The transformer doesn't touch
          the helix.
        </p>

        <p className="mt-3">
          <strong>Cell B: Gemma-4-31B/Latin.</strong> ρ ≈ 0.79. Similar
          to A — the score barely changes between L=0 and peak. CKA ≈ 0.33.
        </p>

        <p className="mt-2 ml-4">
          Reading: same score story, but the integer-to-integer
          relationships have <em>substantially shifted</em>. The
          peak-layer representation arranges the 100 integers into a
          helix, but it's not the same arrangement as the L=0 helix.{" "}
          <strong>Rebuild</strong>. Depth is doing significant
          representational work; it just doesn't show up as a score
          change.
        </p>

        <p className="mt-4">
          <strong>Concrete intuition.</strong> Imagine a 3D helix plot
          at L=0 with integers labelled. Now imagine the 3D helix at
          the peak layer of the same model on the same script.
        </p>

        <ul className="mt-2 list-disc pl-6 space-y-1">
          <li>
            For <strong>Qwen-32B</strong>, the two plots are almost
            identical: integer 23 is in the same position on the helix
            in both, integer 47 is in the same position, etc. Depth
            left the arrangement alone.
          </li>
          <li>
            For <strong>Gemma-31B</strong>, the two plots have the same
            overall helical shape but the labels have moved: integer 23
            might be at the back of the helix at L=0 and at the front
            at peak. The helix is still a helix, but a <em>different</em>{" "}
            one — the row-similarity pattern (which integers are
            nearby which) has changed.
          </li>
        </ul>

        <p className="mt-4">
          <strong>Why linear CKA captures this.</strong> The formula
          (<a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/1905.00414">Kornblith et al. 2019</a>):
        </p>

        <MM>{String.raw`\mathrm{CKA}(X, Y) \;=\; \frac{\,\|X^{\top} Y\|_{F}^{2}\,}{\|X^{\top} X\|_{F} \cdot \|Y^{\top} Y\|_{F}}`}</MM>

        <p>
          (after mean-centring each matrix's columns). Linear CKA
          measures whether the <em>row-similarity pattern</em> in X
          matches the row-similarity pattern in Y. Concretely: take
          the 100×100 matrix <span className="font-mono">XX<sup>⊤</sup></span> —
          entry (i, j) is the inner product between row i and row j.
          That matrix encodes "which integers are close to which" in
          X's feature space. CKA ≈ 1 means this pattern is the same
          for both X and Y.
        </p>

        <p className="mt-3">
          Critically, CKA is <strong>invariant to rotation and
          rescaling of the feature columns</strong>. Rotate Y's
          features and the row-row inner products don't change.
          Rescale them and the formula's normalisation factors out
          the scaling. So CKA isn't asking "are X and Y in the same
          place in residual space?" — they obviously aren't, depth
          has done work. It's asking "is the row-similarity pattern
          the same?" — i.e., are the same integers close to each
          other in both representations? That's the right question
          for "is the shape the same?"
        </p>

        <p className="mt-4">
          <strong>Why helix-projected, not full residual stream.</strong>{" "}
          We compute two flavours:
        </p>

        <ul className="mt-2 list-disc pl-6 space-y-1">
          <li>
            <strong>cka_full</strong> = CKA on the full 4096-d residual
            stream at L=0 vs peak.
          </li>
          <li>
            <strong>cka_helix</strong> = CKA on the helix-projected
            representations <M>{String.raw`B \cdot W^{(L=0)}`}</M> and{" "}
            <M>{String.raw`B \cdot W^{(\text{peak})}`}</M>.
          </li>
        </ul>

        <p className="mt-3">
          The full version mixes in everything: language structure,
          syntactic features, attention metadata, all the things the
          residual stream carries. That's noisy for our question — we
          want to know whether <em>the helix</em> moved, not whether
          the whole model state did. The helix-projected version
          isolates the trig-fittable part of the activations and
          compares only that.
        </p>

        <p className="mt-4">
          <strong>The four quadrants.</strong> Plot ρ on the x-axis
          and CKA on the y-axis. Each (model, script) cell becomes
          one point in the scatter above. Four named regions emerge:
        </p>

        <ul className="mt-2 list-disc pl-6 space-y-1">
          <li>
            <strong>Top-right</strong> (high ρ, high CKA) —{" "}
            <em>pass-through</em>. The input pipeline supplies the
            helix and depth doesn't touch it. Qwen2.5-32B/Latin,
            OLMo/Latin, Pythia × non-Latin positional.
          </li>
          <li>
            <strong>Top-left</strong> (low ρ, high CKA) —{" "}
            <em>depth refines</em>. The L=0 helix is partial; depth
            amplifies its score while preserving its shape.
            Pythia/Latin, Llama/Latin, GPT-J/Latin — K&amp;T's three
            clean cells.
          </li>
          <li>
            <strong>Bottom-right</strong> (high ρ, low CKA) —{" "}
            <em>rebuild</em>. Score barely changes; geometry does.
            Whole Gemma family, Babylonian on Pythia/Llama/OLMo,
            Qwen on Devanagari.
          </li>
          <li>
            <strong>Bottom-left</strong> (low ρ, low CKA) —{" "}
            <em>depth builds new</em>. Score improves <em>and</em>{" "}
            shape changes. Rare. Gemma's strongest non-Latin cells
            edge here.
          </li>
        </ul>

        <p className="mt-4">
          ρ alone collapses top-right and bottom-right into one
          category ("high ρ → inherited"). CKA recovers the
          distinction: top-right is genuine inheritance (depth idle),
          bottom-right is depth doing significant work in a way that
          doesn't move the score. The "Gemma/Latin is mostly
          tokenizer" reading from ρ alone turns out to be wrong once
          CKA is added — Gemma is rebuilding, not inheriting.
        </p>
      </details>
    </section>
  );
}
