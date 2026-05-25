import type { IndexDoc } from "../../lib/types";
import { ClassificationGrid } from "../charts/ClassificationGrid";

interface Props { index: IndexDoc | null; }

/**
 * Slots between WhatTheHelixIs and Finding1Layers. Names the three
 * failure modes of the standard "fit a helix R²" protocol up front,
 * shows a preview classification grid against the 7×8 matrix, and
 * frames the sections that follow as applications of the diagnostic,
 * not standalone findings. Includes a collapsible methods box at the
 * bottom for the precise pipeline.
 */
export function TheDiagnostic({ index }: Props) {
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
          transformer might have done nothing to construct it. You can't
          tell which case you're in without comparing the best layer to
          the very first layer (before any computation). <em>The fix:</em>{" "}
          alongside the peak score, report the ratio
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

      <h3 className="section-subheading">Preview: the matrix, classified</h3>

      <p>
        Applying the joint criteria below (ρ + quality), here's what the
        56 cells look like. The full ρ heatmap and cell-by-cell read are
        in Finding 4 below; this is the headline:
      </p>

      <figure className="my-6">
        <div className="rounded-lg border border-ink/10 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <ClassificationGrid index={index} />
        </div>
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          <strong>Depth-built</strong> = both ρ ≤ 0.55 and helix/PCA ≥ 0.70
          (i.e. the helix is high-quality <em>and</em> substantially
          constructed by transformer depth).{" "}
          <strong>Inherited</strong> = high ρ (the helix score is already
          present at L=0).{" "}
          <strong>Weak</strong> = the trig basis fits little structure at
          the peak. The "narrow truth" the post recovers is the small
          number of teal cells: Pythia/Latin, Llama/Latin, Gemma on the
          scripts it was trained on.
        </figcaption>
      </figure>

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
            tokens by taking the mean. (The alternative is reading only
            the last sub-token, but for multi-digit scripts that bakes
            in a per-digit cycle and produces fake helix peaks; mean is
            the honest default across scripts.)
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
          data (a textbook result called the Eckart–Young theorem). If
          the helix fit is close to this ceiling, the trig basis isn't
          just a good 9-D fit — it <em>is</em> the 9-D structure of the
          activations.
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
          it. For the cell-by-cell classification I use joint criteria:
          a cell counts as <strong>depth-built / depth-amplified</strong>{" "}
          if ρ ≤ 0.55 <em>and</em> the helix/PCA quality ratio is ≥ 0.70;
          as <strong>inherited</strong> if ρ ≥ 0.80; otherwise ambiguous.
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
          random one (same shape, same scale) and re-run only the
          tokenize-and-pool steps. If the random version produces the
          same helix score as the real one, the structure was never in
          the learned weights to begin with. Run on all 12 scripts × 8
          models; see Finding 3 for the comparison. Implemented in{" "}
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
