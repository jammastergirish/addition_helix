/**
 * Top-of-page summary, diagnostic-first. The contribution is the
 * provenance question; the matrix is what was needed to demonstrate
 * the diagnostic. Each bullet is one claim.
 */
export function Findings() {
  return (
    <section className="pt-12 prose-body">
      <h2 className="section-heading">What this shows</h2>
      <ol className="mt-4 space-y-3 list-none pl-0 text-ink-soft">
        <Claim n={1} emphasis>
          A high helix R² is not a provenance measure. The same number can
          come from transformer depth, from the{" "}
          <strong>tokenizer/embedding front end</strong> (rendering +
          tokenization + learned embeddings + pooling), or from a
          basis/window that happened to flatter the data. I introduce a
          simple provenance diagnostic{" "}
          <span className="font-mono">ρ = R²(L=0) / max_L R²(L)</span> and
          pair it with two measurement checks: basis coverage and window
          length.
        </Claim>
        <Claim n={2}>
          The original Latin result survives. Pythia/Latin (ρ = 0.34) and
          Llama/Latin (ρ = 0.39) have most of the helix built by depth —
          these are the cells the original paper studies, and they're the
          cleanest evidence that the helix on those cells is genuinely a
          learned compositional representation.
        </Claim>
        <Claim n={3}>
          Cross-script generalisation is much narrower than the raw helix
          table suggests. Many non-Latin cells fit the trig basis
          non-trivially, but ρ shows most of that structure is already
          present at L=0. Only on Gemma's heavily-trained scripts
          (Devanagari, Persian, Roman) does depth substantially
          contribute outside Latin.
        </Claim>
        <Claim n={4}>
          Babylonian is the cleanest case study. The base-10 basis misses
          base-60 structure; widening the window and adding T=60 recovers
          +8–16 points of R²; ρ shows the recovered structure is mostly
          pre-transformer; and a random-embedding control nails the
          residual — replacing every learned embedding with a random
          vector leaves the L=0 helix essentially unchanged on every
          model (|ΔR²| ≤ 0.007). The geometry is real, but it isn't
          learned numerical representation. It's an emergent property of
          additive symbolic rendering composed with mean-pooling.
        </Claim>
        <Claim n={5} emphasis>
          The general moral: <em>some apparent neural geometry can arise
          from rendering statistics alone</em>, before any learned model
          component runs. Smooth, low-dimensional, Fourier-fittable
          manifolds can be artifacts of how you encode the input — a
          warning for probe work, SAE work, and any "the model represents
          X as a manifold" claim that doesn't include a provenance check.
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
