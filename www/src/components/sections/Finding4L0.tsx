import type { IndexDoc } from "../../lib/types";
import { RhoHeatmap } from "../charts/RhoHeatmap";

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

      <figure className="my-8">
        <div className="rounded-lg border border-ink/10 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <RhoHeatmap index={index} />
        </div>
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          The most consequential row is the bottom of the cool spectrum:
          Pythia and Llama on Latin (ρ = 0.34, 0.39) — the cells the
          original paper studies, and the cleanest evidence that the
          helix on those cells is substantially constructed by depth.
          Everything warm has most of its helix score available before any
          transformer block runs.
        </figcaption>
      </figure>

      <p>
        Reading ρ requires a quality threshold too: a low ρ in a cell
        with a weak peak doesn't mean a hidden depth-built helix, it
        means the trig basis fits little structure anywhere. I'll call a
        cell <strong>depth-built / depth-amplified</strong> if{" "}
        <span className="font-mono">ρ ≤ 0.55</span> and{" "}
        <span className="font-mono">helix/PCA ≥ 0.70</span> at the peak.
        On that joint criterion only four cells of 56 qualify:
        Pythia/Latin, Llama/Latin, and Gemma's two Devanagari cells
        (E4B and 31B). Relaxing helix/PCA to 0.60 adds Gemma's Roman
        cells and Gemma-E4B/Persian; a few Qwen-Roman and Qwen-Devanagari
        cells are borderline (ρ in the 0.67–0.76 range). Either way,
        the bar is "narrow" — not "everything outside Latin is real
        cross-script geometry."
      </p>

      <p>Cell-by-cell:</p>

      <ul className="mt-4 space-y-3 list-disc pl-6">
        <li>
          <strong>Pythia/Latin and Llama/Latin are the paper's clean cells.</strong>{" "}
          ρ = 0.34 and 0.39 — less than half of the helix is at L=0, the
          rest is genuinely built by depth.
        </li>
        <li>
          <strong>Qwen 2.5 takes mode 3 to its limit — even on Latin.</strong>{" "}
          Qwen-7B has ρ = <strong>1.00</strong> (helix R² peaks at L0, the
          embedding output itself), Qwen-32B has ρ = 0.89 (peak at L3 of
          64). Helix R² is healthy on both (~0.54) — the "helix
          replicates on Qwen" reading from Finding 1, read alongside ρ,
          is the cleanest case in the matrix of structure that's the
          tokeniser, not the transformer.
        </li>
        <li>
          <strong>The cross-script extension on Pythia, Llama, and Qwen is
          almost entirely pre-transformer</strong> (ρ = 0.73–1.00 across
          non-Latin cells on those models). Pythia isn't a multilingual
          model; its Arabic-Indic "helix" isn't number representation
          built by the transformer, it's the tokenizer/embedding front
          end: <code>٢٣</code> splits into per-digit sub-tokens whose
          mean varies smoothly with the digit values — exactly the shape
          the trig basis fits.
        </li>
        <li>
          <strong>Gemma genuinely builds the helix on Devanagari, Persian,
          and Roman</strong> (ρ = 0.44–0.65) — the scripts Gemma 4 was
          heavily trained on. The cross-script generalisation that's real
          is training-driven, not architecture-driven. Qwen sits between:
          its only borderline depth-built cells are Roman (ρ ≈ 0.67–0.70)
          and Devanagari on Qwen-32B (ρ ≈ 0.76).
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
          tokenisation artifact at L=0.
        </li>
      </ul>

      <p className="mt-6">
        <strong>The cells where depth genuinely builds the helix are
        narrower than the original headline suggests:</strong> the paper's
        Pythia/Latin and Llama/Latin, plus Gemma's heavily-trained
        scripts. Of 56 cells, 4–6 clear the joint ρ + quality bar
        depending on threshold. Everywhere else, the helix framing
        borrows rhetorical strength from cells where depth did the work
        to talk about cells where it didn't.
      </p>
    </section>
  );
}
