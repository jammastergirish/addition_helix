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
      <p className="mt-2 font-sans text-base md:text-lg text-ink-mute">
        Girish Gupta
      </p>
      <p className="font-sans text-base md:text-sm text-ink-mute"><i>May 2026</i></p>

      <p className="mt-8 text-lg leading-relaxed text-ink-soft">
        Large language models seem to represent numbers as a 3-D spiral
        you can find in their internal activations — a{" "}
        <em>generalised helix</em>, with one straight "number-line"
        direction and circular loops at periods 2, 5, 10, and 100.{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2502.00873">
          Kantamneni &amp; Tegmark (2025)
        </a>{" "}
        found this on three decoder transformers (Pythia-6.9B, GPT-J-6B,
        Llama-3.1-8B), reading Latin digits 0–99. The question I ask
        here is different:{" "}
        <strong>where does that geometry actually come from?</strong>{" "}
        Did the transformer construct it through training, or was it
        already present before any computation — supplied by the input
        pipeline that breaks the number into tokens, looks each token up
        in a learned table, and averages the results?
      </p>

      <p className="mt-4 text-lg leading-relaxed text-ink-soft">
        To answer, I introduce a simple diagnostic — called{" "}
        <strong>ρ</strong> — that compares how well the helix fits at the
        very first layer (before any transformer block has run) to how
        well it fits at the layer where it fits best. If those two scores
        match, the transformer added nothing. If they differ, the
        transformer did real work. I pair ρ with two other checks (is
        the measurement basis right for the script's base; is the input
        range wide enough to see long periods) and apply the kit across{" "}
        <strong>eight models</strong> from six architecture families and{" "}
        <strong>twelve numeral systems</strong> — Latin, Arabic-Indic,
        Persian, Devanagari, Thai, CJK digit string, binary, hexadecimal,
        Greek alphabetic, Hebrew alphabetic, Roman, and Babylonian
        cuneiform.
      </p>

      <p className="mt-4 text-lg leading-relaxed text-ink-soft">
        Kantamneni &amp; Tegmark's Latin results hold up cleanly on all
        three of their models. But most of the "helix-in-script-X-too"
        generalisation you might read off the raw matrix is already
        present <em>before</em> the transformer runs — pre-transformer
        baggage from rendering, tokenization, the learned embedding
        table, and pooling, not something the model has constructed.
      </p>

      <blockquote className="mt-6 border-l-4 border-accent bg-accent/5 px-4 py-3 text-base leading-snug text-ink">
        <strong className="text-accent">The cleanest single result.</strong>{" "}
        On Babylonian cuneiform, swapping every learned embedding for a
        random vector barely changes the helix at the first layer — on
        all eight models, the score gap is{" "}
        <span className="text-ink-mute">≤ 0.007</span>. So that "base-60
        geometry" isn't anything the model learned about cuneiform. It's
        a side-effect of how cuneiform writes numbers: 23 is rendered as
        two ten-wedges plus three one-wedges, and averaging those
        tokens — any tokens, even random ones — gives a vector that
        varies smoothly with the value. <strong>Some apparent neural
        geometry can come from the input pipeline alone, before any
        learning is involved.</strong>
      </blockquote>

      <p className="mt-4 text-sm leading-relaxed text-ink-mute">
        Builds on Kantamneni &amp; Tegmark,{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2502.00873">
          <em>Language Models Use Trigonometry to Do Addition</em>
        </a>{" "}
        (2025), the LLM-scale follow-up to{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2301.05217">
          Nanda et al. (2023)
        </a>{" "}
        (trig basis reverse-engineered in a tiny modular-addition
        transformer; itself an interpretation of{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2201.02177">
          Power et al.'s "grokking"
        </a>). Complementary to{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2605.01148">
          Feucht, Haklay et al. (2026)
        </a>,{" "}
        <em>Arithmetic in the Wild</em>, which identifies a causal
        base-10 addition module in Llama 3.1 8B using the same Fourier
        features — they ask <em>what is the geometry used for</em>; I ask{" "}
        <em>where the geometry came from</em>.
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
