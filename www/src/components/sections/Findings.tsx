/**
 * Top-of-page summary, diagnostic-first. The contribution is the
 * provenance question; the matrix is what was needed to demonstrate
 * the diagnostic. Each bullet is one claim, written for a reader who
 * may not know what "subspace" or "tokenizer" means.
 */
export function Findings() {
  return (
    <section className="pt-12 prose-body">
      <h2 className="section-heading">What this shows</h2>
      <ol className="mt-4 space-y-3 list-none pl-0 text-ink-soft">
        <Claim n={1} emphasis>
          A well-fitting helix doesn't tell us where the helix came from.
          The same fit could mean the transformer built the geometry, or
          that it was already supplied by the input pipeline (rendering,
          tokenization, embedding lookup, pooling), or that the fit just
          happened to flatter the data. So I checked: alongside the helix
          fit at the best layer I also report{" "}
          <strong>ρ</strong>, the ratio of the helix fit at layer 0 to
          the helix fit at its best layer, plus two sanity checks on the
          measurement basis and window.
        </Claim>
        <Claim n={2}>
          The original Latin result survives across all three paper models
          — Pythia/Latin (ρ = 0.34), Llama/Latin (ρ = 0.39), and
          GPT-J/Latin (ρ = 0.41). Low ρ means depth substantially built
          the helix. And <strong>hexadecimal turns out to be a second
          clean case</strong> on the same models (ρ = 0.29–0.35). I
          don't know why; the natural guess is exposure to hex in code
          during pre-training, but that's a hypothesis the data here
          can't test directly.
        </Claim>
        <Claim n={3}>
          Cross-script generalisation is much narrower than the raw helix
          table suggests. Many non-Latin cells fit the trig basis
          non-trivially, but ρ shows the structure was already present
          before the transformer ran. Outside Latin, depth substantially
          contributes only on Gemma's strong non-Latin cells
          (Devanagari, Persian, Roman — plausibly correlated with
          training exposure, though not directly tested here).
        </Claim>
        <Claim n={4}>
          Babylonian cuneiform is the cleanest case of pre-transformer
          structure. The base-10 basis misses base-60 entirely; widening
          the window and adding T=60 to the basis recovers +8–16 points
          of fit; ρ shows the recovered structure is almost all
          pre-transformer; and a random-embedding control nails the
          residual — swap learned embeddings for random vectors and the
          score barely moves on any model (gap ≤ 0.007). The geometry is
          real, but it isn't learned. It's a side-effect of how cuneiform
          writes numbers as wedge counts, averaged together by pooling.
        </Claim>
        <Claim n={5}>
          Two different stories from non-decimal bases.{" "}
          <strong>Babylonian and binary are both mechanical</strong> —
          paper basis hides the structure, native basis recovers it
          (+0.08–0.37 fit), and the random-embedding control shows
          learned and random L=0 are essentially identical. The structure
          is in rendering + pooling alone.{" "}
          <strong>Hex is genuinely depth-built</strong>: hex digit
          embeddings carry learned structure (random-embed gap {">"} 0 on
          every model), and depth roughly doubles the score on
          Pythia/Llama/GPT-J/OLMo (ρ = 0.29–0.35). <em>Each base needs
          its own protocol fix before its ρ becomes interpretable.</em>
        </Claim>
        <Claim n={6} emphasis>
          The general lesson: <em>some apparent neural geometry can come
          from rendering statistics alone</em> — before learned
          embeddings or transformer blocks are needed. Smooth,
          low-dimensional, easy-to-fit manifolds
          can be artifacts of how you encode the input — a warning for
          probe work, sparse-autoencoder work, and any "the model
          represents X as a manifold" claim that doesn't include a
          provenance check.
        </Claim>
        <Claim n={7}>
          A second check — <strong>CKA</strong> (how similar the
          integer-to-integer geometry at the first layer is to the
          geometry at the best layer) — sharpens the picture further.
          "High ρ" can mean either real pass-through (Qwen-32B/Latin:
          CKA ≈ 0.9, depth touches the geometry barely) or
          rebuild-with-same-score (Gemma/Latin: CKA ≈ 0.35, depth
          produces a similar-quality helix with a substantially
          different integer-to-integer pattern). Two mechanistically
          distinct cases ρ alone would lump together.
        </Claim>
      </ol>
    </section>
  );
}

function Claim({ n, emphasis, children }: { n: number; emphasis?: boolean; children: React.ReactNode }) {
  return (
    <li
      className={[
        "relative pl-12 leading-relaxed",
        emphasis
          ? "border-l-2 border-accent pl-10 -ml-2 bg-paper-warm/40 py-3 pr-3 rounded-r"
          : "",
      ].join(" ")}
    >
      <span className="absolute left-0 top-1 inline-flex h-7 w-7 items-center justify-center rounded-full bg-ink/5 font-mono text-[12px] text-ink/70">
        {n}
      </span>
      {children}
    </li>
  );
}
