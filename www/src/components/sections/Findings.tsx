/**
 * Top-of-page summary, diagnostic-first. The contribution is the
 * measurement framework; the matrix is what was built to demonstrate
 * it. Each bullet is one claim.
 */
export function Findings() {
  return (
    <section className="pt-12 prose-body">
      <h2 className="section-heading">What this shows</h2>
      <ol className="mt-4 space-y-3 list-none pl-0 text-ink-soft">
        <Claim n={1} emphasis>
          The standard "fit a helix R²" protocol has <em>three</em> failure
          modes that systematically inflate the reported result — basis
          bandwidth, window wrap, and (most consequentially) tokeniser
          inheritance via the L=0 / peak ratio ρ.
        </Claim>
        <Claim n={2}>
          I built a <strong>7 models × 8 numeral systems</strong> matrix —
          56 cells — to surface them. Of the 56, only ~5 are cleanly
          depth-built helices: Pythia/Latin, Llama/Latin, and Gemma on the
          scripts it was heavily trained on.
        </Claim>
        <Claim n={3}>
          A naive read of the matrix would have given the headline "the
          helix generalises across positional scripts (Devanagari,
          Arabic-Indic, Chinese) and even to base-60 Babylonian on every
          model." The ρ diagnostic shows almost all of that is the
          tokeniser absorbing numeric semantics during pre-training, not
          the transformer computing with it.
        </Claim>
        <Claim n={4}>
          Feucht &amp; Haklay (May 2026) recently confirmed the helix{" "}
          <em>is</em> causally read on Llama/Latin — exactly the cell ρ
          marks as depth-built. The matrix tells them which cells are
          worth that causal investigation. Two halves of the same
          diagnostic kit.
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
