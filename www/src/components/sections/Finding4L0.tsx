import type { IndexDoc } from "../../lib/types";
import { RhoHeatmap } from "../charts/RhoHeatmap";

interface Props { index: IndexDoc | null; }

export function Finding4L0({ index }: Props) {
  return (
    <section className="prose-body">
      <h2 className="section-heading">
        But most of what we just reported is the tokeniser, not the model
      </h2>

      <p>
        Every numeral's residual stream at layer 0 — before any transformer
        block has run — is just <code>mean(embed(sub_tokens(a)))</code>:
        pure tokeniser plus pooling. If the trig basis already fits{" "}
        <em>that</em>, the helix we report at the peak layer was mostly
        inherited from the embedding lookup, not built by depth.
      </p>

      <p>
        Define{" "}
        <span className="font-mono">ρ = helix R²(L=0) / helix R²(peak)</span>.{" "}
        <strong>ρ ≈ 1</strong>: the transformer adds nothing — the helix is
        a tokeniser artifact. <strong>ρ small</strong>: depth is doing the
        real work.
      </p>

      <figure className="my-8">
        <div className="rounded-lg border border-ink/10 bg-white p-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <RhoHeatmap index={index} />
        </div>
        <figcaption className="mx-auto mt-3 max-w-prose text-sm text-ink-mute leading-snug">
          ρ for every (model, script). Cool colours = depth builds it. Warm
          colours = tokeniser preserves it.
        </figcaption>
      </figure>

      <p>This reframes most of what we just reported:</p>

      <ul className="mt-4 space-y-3 list-disc pl-6">
        <li>
          <strong>Pythia/Latin and Llama/Latin are still clean wins</strong>{" "}
          for the original paper: ρ = 0.34 and 0.39 — less than half the
          helix is at L=0, the rest is genuinely built by depth.
        </li>
        <li>
          <strong>The cross-script extension on Pythia and Llama is almost
          entirely tokeniser</strong> (ρ = 0.73–1.00). Pythia isn't a
          multilingual model; its Arabic-Indic "helix" isn't number
          representation, it's the tokenizer splitting{" "}
          <code>٢٣</code> into per-digit sub-tokens whose mean varies
          smoothly with the digit values.
        </li>
        <li>
          <strong>Gemma does build the helix on Devanagari, Persian, and
          Roman</strong> (ρ = 0.44–0.65) — the scripts Gemma 4 was heavily
          trained on. The cross-script generalisation that's real is
          training-driven, not architecture-driven.
        </li>
        <li>
          <strong>Babylonian's base-60 is universally tokeniser-driven</strong>{" "}
          (ρ = 0.82–0.89 on every model). The wedge embeddings carry value
          information from "𒁹 means one"-style text; mean-pooling produces
          the additive shape the basis fits. Depth adds ~15% on top. The
          base-60 finding is real but is mostly about the embedding layer.
        </li>
        <li>
          <strong>Greek has no helix anywhere</strong> (ρ = 1.00 on Pythia,
          Llama, OLMo). Greek alphabetic numerals are treated as ordinary
          short letter-sequences with no numeric semantics; what little
          structure the basis fits is a tokenisation artifact at L=0.
        </li>
      </ul>

      <p className="mt-6">
        <strong>The cells where depth genuinely builds the helix are
        narrower than the original headline suggests:</strong> the paper's
        Pythia/Latin and Llama/Latin, plus Gemma on the scripts it was
        trained on. Everywhere else, the helix framing borrows rhetorical
        strength from cells where depth did the work to talk about cells
        where it didn't.
      </p>

      <h3 className="section-subheading">Three ways the standard protocol can mislead</h3>
      <p>
        The L=0 vs peak check is the third of three places the paper's
        original measurement protocol can produce a result that's an
        artifact of how we measured, not of what the model does. The other
        two are visible elsewhere in this experiment:
      </p>

      <ol className="mt-4 space-y-3 list-decimal pl-6">
        <li>
          <strong>The basis has bandwidth.</strong> Helix R² is a regression
          fit to a fixed set of periods. If the model encodes something at
          a period the basis doesn't include, the metric reads low even when
          the structure is strong. Same data, same model: <em>+0.13 helix R²
          just from adding T=60 to the basis</em> (Babylonian). The fix is
          to cross-check the basis fit against the basis-free FFT — if a
          peak appears at a period not in the basis, helix R² is
          undermeasuring; add the period and refit.
        </li>
        <li>
          <strong>The window must wrap.</strong> To detect period{" "}
          <span className="font-mono">T</span> via FFT you need several
          wraps of <span className="font-mono">T</span> in the input range.
          The default 0..99 is fine for T = 2, 5, 10 (50, 20, 10 wraps),
          barely workable for T = 100 (one wrap), and <em>not</em> workable
          for T = 60 (one wrap, phase-misaligned). The fix is to widen the
          window to many wraps of any candidate period.
        </li>
        <li>
          <strong>The tokeniser inherits.</strong> A high helix R² at the
          peak layer can be entirely inherited from the L=0 embedding mean,
          and reads identically to one the model built — this whole
          section. Reporting ρ alongside the peak is the cheapest way to
          tell which is which.
        </li>
      </ol>

      <p className="mt-4">
        The first two we addressed with explicit measurement fixes (and
        gained the Babylonian base-60 finding). The third is what reframes
        most of the rest.
      </p>
    </section>
  );
}
