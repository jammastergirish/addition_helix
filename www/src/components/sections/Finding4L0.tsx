import type { IndexDoc } from "../../lib/types";
import { RhoHeatmap } from "../charts/RhoHeatmap";
import { ClassificationGrid } from "../charts/ClassificationGrid";

interface Props { index: IndexDoc | null; }

export function Finding4L0({ index }: Props) {
  return (
    <section className="prose-body">
      <h2 className="section-heading">
        The ρ heatmap, cell by cell
      </h2>

      <p>
        ρ for every (model, script) at paper-default settings (n = 100,
        basis [2, 5, 10, 100]). Cool colours = depth substantially adds
        helix score. Warm colours = helix score was already present at L=0.
      </p>

      <p className="my-3 rounded border-l-4 border-amber-400 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        <strong>Read this with care.</strong> This heatmap shows
        <em> paper-default</em> ρ. Binary, hexadecimal, and Babylonian
        are intentionally mismeasured here — their natural periods don't
        fit the paper basis, and their windows are too short. Their
        interpretable ρ values come from the native-basis runs in
        Part 4 (the two case-study sections). Don't read the
        binary/hex/Babylonian cells off this heatmap as a final
        verdict.
      </p>

      <figure className="my-8">
        <div className="rounded-lg border border-ink/10 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <RhoHeatmap index={index} />
        </div>
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          The most consequential row is the bottom of the cool spectrum:
          Pythia, Llama, and GPT-J on Latin (ρ = 0.34, 0.39, 0.41) — the
          cells the original paper studies, and the cleanest evidence
          that the helix on those cells is substantially constructed by
          depth. Everything warm has most of its helix score available
          before any transformer block runs.
        </figcaption>
      </figure>

      <p>
        Reading ρ requires a quality threshold too: a low ρ in a cell
        with a weak peak doesn't mean a hidden depth-built helix, it
        means the trig basis fits little structure anywhere. I'll call a
        cell <strong>depth-built / depth-amplified</strong> if{" "}
        <span className="font-mono">ρ ≤ 0.55</span> and{" "}
        <span className="font-mono">helix/PCA ≥ 0.70</span> at the peak.
        On that joint criterion six cells of 96 qualify: the three Latin
        cells from the original paper (Pythia, Llama, GPT-J), Gemma's two
        Devanagari cells (E4B and 31B), and Gemma-31B/Binary — though
        Binary at paper defaults is one of the cells the heatmap warns is
        mismeasured, so I'd usually set it aside and read its native-basis
        ρ in the case studies. Relaxing helix/PCA to 0.60 adds Gemma's
        two Roman cells; a few Qwen-Roman and Qwen-Devanagari cells are
        borderline (ρ in the 0.67–0.77 range). Either way, the bar is
        "narrow" — not "everything outside Latin is real cross-script
        geometry."
      </p>

      <p>Cell-by-cell:</p>

      <ul className="mt-4 space-y-3 list-disc pl-6">
        <li>
          <strong>The three paper-model Latin cells are clean.</strong>{" "}
          Pythia/Latin ρ = 0.34, Llama/Latin ρ = 0.39, GPT-J/Latin ρ =
          0.41 — less than half of the helix is at L=0, the rest is
          genuinely built by depth.
        </li>
        <li>
          <strong>Qwen 2.5 takes mode 3 to its limit — even on Latin.</strong>{" "}
          Qwen-7B has ρ = <strong>1.00</strong> (helix R² peaks at L0, the
          embedding output itself), Qwen-32B has ρ = 0.89 (peak at L3 of
          64). Helix R² is healthy on both (~0.54) — the "helix
          replicates on Qwen" reading from Finding 1, read alongside ρ,
          is the cleanest case in the matrix of structure that's the
          tokenizer, not the transformer.
        </li>
        <li>
          <strong>The cross-script extension on Pythia and Llama is
          almost entirely pre-transformer</strong> (ρ = 0.73–1.00 across
          non-Latin positional cells on those two models; Qwen is broken
          out below). Pythia isn't a multilingual
          model; its Arabic-Indic "helix" isn't number representation
          built by the transformer, it's the tokenizer/embedding front
          end: <code>٢٣</code> splits into per-digit sub-tokens whose
          mean varies smoothly with the digit values — exactly the shape
          the trig basis fits.
        </li>
        <li>
          <strong>Gemma genuinely builds the helix on Devanagari, Persian,
          and Roman</strong> (ρ = 0.44–0.63). These are scripts where
          Gemma 4 plausibly had substantial training exposure, though
          I don't measure pre-training distributions directly. Qwen
          sits between: its only borderline depth-built cells are
          Roman (ρ ≈ 0.67–0.70) and Devanagari (ρ ≈ 0.76–0.77).
        </li>
        <li>
          <strong>Babylonian is universally pre-transformer</strong>{" "}
          (ρ = 0.82–0.93 on every model). The base-60 finding from the
          previous section is real but mostly about the embedding layer
          — and possibly mechanical (renderer + pooling); see the
          random-embedding control in Finding 3.
        </li>
        <li>
          <strong>Greek has no helix anywhere</strong> (ρ ≥ 0.94 on every
          model that doesn't actively train on Greek). Greek alphabetic
          numerals are treated as ordinary short letter-sequences with no
          numeric semantics; what little structure the basis fits is a
          tokenization artifact at L=0.
        </li>
      </ul>

      <p className="mt-6">
        <strong>The cells where depth genuinely builds the helix are
        narrower than the original headline suggests:</strong> the three
        paper-model Latin cells, plus Gemma's strong non-Latin scripts.
        Of 96 cells, 6–8 clear the joint ρ + quality bar depending on
        threshold. Everywhere else, the helix framing borrows rhetorical
        strength from cells where depth did the work to talk about cells
        where it didn't.
      </p>

      <h3 className="section-subheading">The classification grid</h3>

      <p>
        Applying the mutually exclusive criteria from the Diagnostic
        kit (depth-built / depth-amplified / inherited / weak /
        ambiguous), here's the same 96-cell matrix coloured by class:
      </p>

      <figure className="my-6">
        <div className="rounded-lg border border-ink/10 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <ClassificationGrid index={index} />
        </div>
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          <strong>Depth-built</strong> = ρ low AND helix high-quality.{" "}
          <strong>Depth-amplified</strong> = partial depth contribution.{" "}
          <strong>Inherited</strong> = high ρ at decent quality
          (pre-transformer). <strong>Weak</strong> = the basis fits
          little. <strong>Ambiguous</strong> = between thresholds. The
          "narrow truth" of the post is the small number of teal cells.
        </figcaption>
      </figure>
    </section>
  );
}
