/**
 * Top-of-page summary. Four claims, one sentence each. The fourth is the
 * tease for the section that overturns the first three.
 */
export function Findings() {
  return (
    <section className="pt-12 prose-body">
      <h2 className="section-heading">What this shows</h2>
      <ol className="mt-4 space-y-3 list-none pl-0 text-ink-soft">
        <Claim n={1}>
          The helix is real and replicates in two new architecture families
          — but where it lives in the stack varies wildly across models.
        </Claim>
        <Claim n={2}>
          It generalises across positional numeral scripts (Devanagari,
          Arabic-Indic, Chinese) — but only on models that were actually
          trained on them.
        </Claim>
        <Claim n={3}>
          Even Babylonian cuneiform encodes clean base-60 structure, on
          every model — once you stop measuring with a base-10 ruler.
        </Claim>
        <Claim n={4} emphasis>
          But on most cells we tested, what we're measuring is the
          tokeniser, not the transformer — the helix is already present in
          the embeddings before any block has run.
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
