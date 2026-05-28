import { M } from "../Math";

export function Hero() {
  return (
    // Same layout grid as <main>: every direct child lands in the `main`
    // column, so the hero text aligns exactly with the body text below.
    <header className="layout border-b border-ink/10 pt-16 pb-10 md:pt-24 md:pb-14">
      {/* <div className="mb-6 flex flex-wrap items-center gap-2 text-xs uppercase tracking-wider text-ink-mute">
        <span className="pill border-ink/15 text-ink-mute">Mechanistic Interp</span>
        <span className="pill border-accent/40 text-accent">5 models · 8 scripts</span>
      </div> */}

      <h1 className="font-sans text-3xl md:text-5xl font-semibold tracking-tight leading-[1.1]">
        From Latin Digits to Babylonian Cuneiform:{" "}
        <span className="text-accent">Measuring the Provenance of Number Helices in LLMs</span>
      </h1>
      <p className="mt-4 font-serif text-lg md:text-xl italic text-ink-soft leading-snug">
        A high helix score is not enough to show the transformer built
        the geometry.
      </p>
      <p className="mt-4 font-sans text-base md:text-lg text-ink-mute">
        Girish Gupta
      </p>
      <p className="font-sans text-base md:text-sm text-ink-mute"><i>May 2026</i></p>

      <p className="mt-8 text-lg leading-relaxed text-ink-soft">
        Large language models seem to represent integers as generalised
        helices in residual space: one straight "number-line" direction
        plus circular features at periods such as 2, 5, 10, and 100.{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2502.00873">
          Kantamneni &amp; Tegmark (2025)
        </a>{" "}
        found this on Pythia-6.9B, GPT-J-6B, and Llama-3.1-8B, reading
        Latin digits 0–99. But a high helix score doesn't tell us where
        the geometry came from. It may be{" "}
        <strong>constructed by transformer depth</strong>, or{" "}
        <strong>inherited from the input pipeline</strong> (rendering,
        tokenization, embedding lookup, pooling), or{" "}
        <strong>induced by the measurement protocol itself</strong>{" "}
        (wrong basis, too-short input range).
      </p>

      <p className="mt-4 text-lg leading-relaxed text-ink-soft">
        This provenance question is also motivated by prior behavioral
        work on number tokenization.{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2402.14903">
          Singh &amp; Strouse (2024)
        </a>{" "}
        show that arithmetic performance in frontier LLMs depends
        strongly on how numbers are tokenized: changing the tokenization
        direction can substantially improve GPT-3.5/4 arithmetic, and
        the resulting errors follow systematic token-boundary patterns.
        Their result is behavioral; the question here is
        representational. If tokenization can change arithmetic
        behavior, can the input pipeline also create the internal
        numerical geometry we later interpret as a model representation?
      </p>

      <p className="mt-4 text-lg leading-relaxed text-ink-soft">
        To ask where the geometry comes from, I report four
        measurements per cell: a provenance ratio{" "}
        <M>{String.raw`\rho = R^2_{\mathrm{helix}}(L{=}0) \,/\, \max_{L}\, R^2_{\mathrm{helix}}(L)`}</M>;
        a basis/window check; a random-embedding control; and a
        representation-alignment test (CKA), across{" "}
        <strong>eight models</strong> and{" "}
        <strong>twelve numeral systems</strong>.
      </p>

      <p className="mt-4 text-lg leading-relaxed text-ink-soft">
        Kantamneni &amp; Tegmark deliberately avoided many
        tokenization complications by focusing on space-prefixed
        Latin integers that are single tokens in their models — the
        right clean setting for their causal addition study. My
        question starts where that simplification stops: when we
        extend the helical fit across scripts and bases, the input
        pipeline itself can create or preserve Fourier-fittable
        structure, so provenance checks become necessary. K&amp;T's
        Latin results hold up cleanly on all three of their models;
        but most of the "helix-in-script-X-too" generalisation you
        might read off the raw matrix is already present{" "}
        <em>before</em> the transformer runs — pre-transformer baggage
        from rendering, tokenization, the embedding table, and pooling,
        not something the model has constructed.
      </p>

      <blockquote className="mt-6 border-l-4 border-accent bg-accent/5 px-4 py-3 text-base leading-snug text-ink">
        <strong className="text-accent">The cleanest single result.</strong>{" "}
        On Babylonian cuneiform, swapping every learned embedding for a
        random vector barely changes the helix at the first layer — on
        all eight models, the score gap is{" "}
        <span className="text-ink-mute">≤ 0.007</span>. So the measured
        base-60 geometry at layer 0 isn't evidence of learned cuneiform
        semantics. It arises mechanically: from how cuneiform renders
        numbers as wedge counts, how the tokenizer splits those wedges,
        and how mean-pooling combines them. <strong>Some apparent neural
          geometry can come from the input pipeline alone, before learned
          embeddings or transformer blocks are needed.</strong>
      </blockquote>

      <p className="mt-4 text-sm leading-relaxed text-ink-mute">
        Background: the trig basis was first reverse-engineered by{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2301.05217">
          Nanda et al. (2023)
        </a>{" "}
        in a tiny modular-addition transformer — itself an
        interpretation of{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2201.02177">
          Power et al.'s "grokking"
        </a>. Complementary to{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2605.01148">
          Feucht, Haklay et al. (2026)
        </a>,{" "}
        <em>Arithmetic in the Wild</em>, which identifies a causal
        base-10 addition module in Llama 3.1 8B using the same Fourier
        features — they ask <em>what is the geometry used for</em>;
        I ask <em>where the geometry came from</em>.
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-ink-mute">
        <span>Pythia-6.9B</span>
        <span className="text-ink/20">·</span>
        <span>GPT-J-6B</span>
        <span className="text-ink/20">·</span>
        <span>Llama-3.1-8B</span>
        <span className="text-ink/20">·</span>
        <span>Gemma-4-E4B</span>
        <span className="text-ink/20">·</span>
        <span>Gemma-4-31B</span>
        <span className="text-ink/20">·</span>
        <span>OLMo-3-32B</span>
        <span className="text-ink/20">·</span>
        <span>Qwen2.5-7B</span>
        <span className="text-ink/20">·</span>
        <span>Qwen2.5-32B</span>
      </div>
    </header>
  );
}
