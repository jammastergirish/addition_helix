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
        Large language models encode integers in their residual stream as a{" "}
        <em>generalised helix</em> — one linear "number-line" axis plus
        modular circles at period 2, 5, 10, and 100. But a high helix R²
        does not, by itself, tell us where that geometry came from: the
        transformer may have built it, the{" "}
        <strong>tokenizer/embedding front end</strong> may already supply
        it, or the measurement basis may have flattered (or missed) the
        underlying structure. I introduce a provenance diagnostic{" "}
        <span className="font-mono">ρ = R²(L=0) / max_L R²(L)</span> and
        apply it, alongside two basis/window checks and a random-embedding
        control, across <strong>eight models</strong> from six architecture
        families and <strong>eight numeral systems</strong> — Latin,
        Arabic-Indic, Persian, Devanagari, positional Chinese (CJK digit
        strings), Greek alphabetic, Roman, and Babylonian cuneiform. The
        Pythia/Latin and Llama/Latin results survive cleanly. Most apparent
        cross-script generalisation is already present at L=0 —
        pre-transformer inheritance from rendering, tokenization, learned
        embeddings, and pooling — rather than constructed by transformer
        depth.
      </p>

      <blockquote className="mt-6 border-l-4 border-accent bg-accent/5 px-4 py-3 text-base leading-snug text-ink">
        <strong className="text-accent">The cleanest single result.</strong>{" "}
        On Babylonian cuneiform, replacing every learned embedding with a
        random vector leaves the L=0 base-60 helix essentially unchanged
        across all eight models{" "}
        <span className="text-ink-mute">(|ΔR²| ≤ 0.007)</span>. The
        apparent Fourier geometry of cuneiform numerals is not learned
        numerical representation — it's an emergent property of additive
        symbolic rendering composed with mean-pooling. Some apparent
        neural geometry can arise from rendering statistics alone, before
        any learned component runs.
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
