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
        <span className="text-accent">Provenance and Measurement in Numerical Geometry of LLMs</span>
      </h1>
      <p className="mt-2 font-sans text-base md:text-lg text-ink-mute">
        Girish Gupta
      </p>
      <p className="font-sans text-base md:text-sm text-ink-mute"><i>May 2026</i></p>

      <p className="mt-8 text-lg leading-relaxed text-ink-soft">
        Large language models encode integers in their residual stream as a{" "}
        <em>generalised helix</em> — one linear "number-line" axis plus
        modular circles at period 2, 5, 10, and 100. The standard "fit a
        helix R²" measurement protocol has three failure modes that
        systematically inflate the reported result; the most consequential
        is <strong>tokeniser inheritance</strong> — the helix R² at the
        peak layer can be entirely supplied by the L=0 embeddings and read
        identically to one the transformer built. To surface this, I
        applied the protocol with a new diagnostic across{" "}
        <strong>seven models</strong> from five architecture families and{" "}
        <strong>eight numeral systems</strong> — Latin, Arabic-Indic,
        Persian, Devanagari, positional Chinese, Greek alphabetic, Roman,
        and Babylonian cuneiform. Of 56 cells, only ~5 are cleanly
        depth-built helices. Most of what reads as cross-script
        generalisation is the tokeniser, not the transformer.
      </p>

      <p className="mt-4 text-sm leading-relaxed text-ink-mute">
        A from-scratch extension of Kantamneni &amp; Tegmark,{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2502.00873">
          <em>Language Models Use Trigonometry to Do Addition</em>
        </a>{" "}
        (2025) — itself the LLM-scale follow-up to{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2301.05217">
          Nanda et al. (2023)
        </a>, who first reverse-engineered the same trig basis in a
        small transformer trained from scratch on modular addition. That
        was the original "grokking" task of{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2201.02177">
          Power et al. (2022)
        </a>, where a small network trained on modular arithmetic was
        famously left running long past memorisation and suddenly
        generalised.
      </p>

      <p className="mt-3 text-sm leading-relaxed text-ink-mute">
        Complementary to{" "}
        <a className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" href="https://arxiv.org/abs/2605.01148">
          Feucht, Haklay et al. (2026)
        </a>,{" "}
        <em>Arithmetic in the Wild</em>, which identifies a causal
        base-10 addition module in Llama 3.1 8B using exactly these
        Fourier features — the "what is the geometry used for" question
        this post measures the geometry of.
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-ink-mute">
        <span>Pythia-6.9B</span>
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
