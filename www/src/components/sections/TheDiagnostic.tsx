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
        The standard protocol — fit a 9-feature trig basis (one linear
        column, four (cos, sin) pairs at <span className="font-mono">T ∈ {"{"}2, 5, 10, 100{"}"}</span>) to a
        residual stream of 100 integers, read at one layer — was designed
        for Pythia and Llama on Latin. Apply it anywhere else and three
        failure modes can make the protocol answer the wrong question.{" "}
        <strong>Two of them hide real structure; one misattributes
        inherited structure as depth-built geometry.</strong>
      </p>

      <ol className="mt-6 space-y-4 list-decimal pl-6">
        <li>
          <strong>The basis has bandwidth (under-detection).</strong>{" "}
          Helix R² is a regression fit to a fixed set of periods. If the
          model encodes something at a period the basis doesn't include,
          the metric reads low even when the structure is strong.{" "}
          <em>The fix:</em> cross-check against the basis-free FFT and add
          missing periods. (Babylonian gains 8–16 percentage points just
          from adding T=60.)
        </li>
        <li>
          <strong>The window must wrap (under-detection).</strong> To
          detect period <span className="font-mono">T</span> via FFT you
          need several wraps of <span className="font-mono">T</span> in
          the input range. The default 0..99 is fine for T = 2, 5, 10
          (50, 20, 10 wraps), barely workable for T = 100 (one wrap), and{" "}
          <em>not</em> workable for T = 60 (one wrap, phase-misaligned).{" "}
          <em>The fix:</em> widen the window to many wraps of any
          candidate period.
        </li>
        <li>
          <strong>The tokenizer/embedding front end inherits (misattribution).</strong>{" "}
          A high helix R² at the peak layer can be entirely supplied by
          the L=0 representation — itself the composition of string
          rendering, BPE tokenization, learned embedding lookup, and
          mean-pooling. It reads <em>identically</em> to a helix the
          transformer built. <em>The fix:</em> alongside the peak, report
          the provenance ratio
        </li>
      </ol>

      <p className="mt-3 ml-12 font-mono text-sm text-ink/80">
        ρ = R²(L=0) / max<sub>L</sub> R²(L).
      </p>

      <p className="mt-3 ml-12">
        ρ ≈ 1 means depth did not improve the helix score. This is a{" "}
        <em>provenance diagnostic</em>, not a causal proof: it does not
        prove the transformer never reads the geometry — only that the
        geometry was already available at L=0. Distinguishing
        depth-amplification (similar score, different subspace) from
        pure inheritance would need an additional subspace-alignment
        check (CKA, Procrustes), which I do not run here.
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
          <strong>Pipeline.</strong> For each (model, script, integer{" "}
          <span className="font-mono">a</span>):
        </p>
        <ol className="mt-1 list-decimal pl-6 space-y-1">
          <li>
            Render <span className="font-mono">a</span> as a string in the
            chosen numeral system (see "The twelve numeral systems" above).
          </li>
          <li>
            Tokenize with the model's own tokenizer (BPE for all eight
            models tested), without prompt scaffolding.
          </li>
          <li>
            Forward-pass the model with{" "}
            <span className="font-mono">output_hidden_states=True</span>,
            capturing every layer.
          </li>
          <li>
            <strong>Pool</strong> across the numeral's sub-tokens by mean
            (<span className="font-mono">--pool mean</span>). The
            alternative <span className="font-mono">--pool last</span>{" "}
            reads only the last sub-token and bakes a per-digit cycle into
            the data for multi-digit scripts; mean is the honest
            cross-script default.
          </li>
          <li>
            Stack into{" "}
            <span className="font-mono">H ∈ R^(N × d)</span> per layer (N
            = 100 by default; 600 for Babylonian wide-window analyses).
          </li>
        </ol>

        <p className="mt-3">
          <strong>Layer indexing.</strong> Hidden states are 0-indexed:{" "}
          <span className="font-mono">L=0</span> is the embedding output
          (token embedding + positional if any, before any transformer
          block); <span className="font-mono">L=k</span> for{" "}
          <span className="font-mono">k≥1</span> is the residual stream
          after block <span className="font-mono">k</span>. For a model
          with <span className="font-mono">N_layers</span> blocks,{" "}
          <span className="font-mono">hidden_states</span> has length{" "}
          <span className="font-mono">N_layers + 1</span>.
        </p>

        <p className="mt-3">
          <strong>Basis fit.</strong> The trig basis is{" "}
          <span className="font-mono">B(a) = [a, cos(2πa/T), sin(2πa/T)]</span>{" "}
          for <span className="font-mono">T ∈ {"{"}2, 5, 10, 100{"}"}</span>{" "}
          (one linear + 4 × 2 = 9 features). Fit by ordinary least
          squares: <span className="font-mono">W = argmin ‖H − BW‖_F²</span>.
          Reported helix R² is variance-weighted across output dims. The
          PCA upper bound is the best 9-D reconstruction R² (Eckart–Young).
        </p>

        <p className="mt-3">
          <strong>ρ definition.</strong>{" "}
          <span className="font-mono">ρ = R²_helix(L=0) / max_L R²_helix(L)</span>.
          For thresholds used in the cell-by-cell analysis: I treat
          ρ ≤ 0.55 with helix/PCA ≥ 0.70 as "depth-built / depth-amplified,"
          ρ ≥ 0.80 as "embedding-inherited," and the rest as ambiguous.
        </p>

        <p className="mt-3">
          <strong>Zero handling.</strong> Positional base-10 scripts
          (Latin, Arabic-Indic, Persian, Devanagari, CJK) render 0
          standardly. Non-positional systems lack a native zero: I use{" "}
          <span className="font-mono">"nulla"</span> (Roman),{" "}
          <span className="font-mono">"Ø"</span> (Greek), and U+1244A{" "}
          (Babylonian late-period zero) as placeholders. Headline ρ and
          helix-R² values are robust to dropping <span className="font-mono">a=0</span>.
        </p>

        <p className="mt-3">
          <strong>Random-embedding control.</strong> For every (model,
          script) combination I additionally generate a random embedding
          matrix of the model's dimensions (<span className="font-mono">N(0, 1/√d)</span>),
          look up token IDs through it instead of the learned table, and
          mean-pool. If the random-embedding L=0 R² ≈ the learned L=0
          R², the L=0 structure is mechanical (renderer + tokenization +
          pooling), not learned semantics. Run for all 12 scripts × 8
          models; see Finding 3 for the comparison. Implemented in{" "}
          <span className="font-mono">embed_control.py</span>.
        </p>

        <p className="mt-3">
          <strong>Subspace alignment (CKA).</strong> Linear CKA between
          H[L=0] and H[peak_layer], and between their projections onto
          the fitted helix basis (B · W). Distinguishes "depth preserves"
          from "depth rebuilds" when ρ alone is ambiguous. Currently run
          on Latin × all 8 models; see Finding 5. Implemented in{" "}
          <span className="font-mono">subspace_align.py</span>.
        </p>

        <p className="mt-3">
          <strong>What's <em>not</em> here.</strong> No causal
          intervention; no held-out cross-validation of the basis fit;
          only one prompt template per script; subspace alignment run
          only on Latin (extension to non-Latin cells is straightforward
          but requires re-running each model).
        </p>
      </details>
    </section>
  );
}
