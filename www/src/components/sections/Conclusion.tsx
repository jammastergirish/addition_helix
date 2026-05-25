export function Conclusion() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">What this changes</h2>

      <p>
        The contribution here is the diagnostic kit, not the matrix.
        The standard "fit a helix and report R²" procedure has three
        problems that can make it answer the wrong question: two hide
        real structure (wrong basis, too-short window) and one credits
        the transformer with structure it didn't build (which ρ catches).
        Adding CKA turns up a fourth nuance: a high ρ doesn't always
        mean the transformer left the geometry alone — sometimes it
        rebuilds the integer-to-integer pattern with a substantially
        different shape but ends up with a similar score. Putting all
        four checks together across the matrix gives a four-way
        classification:
      </p>

      <ul className="mt-3 space-y-2 list-disc pl-6">
        <li>
          <strong>Pass-through cells</strong> (high ρ AND high CKA): the
          input pipeline supplies the helix and depth doesn't touch it.
          Qwen-32B/Latin, OLMo/Latin, most Pythia/Llama non-Latin
          positional cells, Qwen on Chinese/Greek. The clearest existence
          proof of pre-transformer inheritance — Qwen 2.5/Latin most
          sharply (helix R² ≈ 0.54, ρ ≈ 1, CKA ≈ 1).
        </li>
        <li>
          <strong>Depth-amplification cells</strong> (low-to-moderate ρ
          AND high CKA): the L=0 helix is real and depth refines it.
          Pythia/Latin, Llama/Latin, GPT-J/Latin, Roman on almost every
          model, and <strong>hexadecimal on Pythia/Llama/GPT-J/OLMo</strong>{" "}
          — hex is the second-cleanest depth-built case in the matrix;
          the natural guess is hex exposure in pre-training (code,
          addresses, hashes), though that's not directly tested here.
        </li>
        <li>
          <strong>Mechanical cells</strong> (learned-vs-random gap ≈ 0
          at L=0): Babylonian on every model, <em>and</em> binary on
          every model. The structure comes from rendering + tokenization
          + pooling statistics alone — symbol counts (wedges, bits)
          turned into vector arithmetic by mean-pooling, with or without
          learned embeddings.
        </li>
        <li>
          <strong>Rebuild cells</strong> (high ρ but LOW CKA): depth
          produces a similar-score helix with a substantially different
          integer-to-integer pattern, flattening to "inherited" under ρ
          alone. The whole Gemma family on most scripts, Babylonian on
          Pythia/Llama/OLMo, Qwen on Devanagari/Thai. Gemma/Latin is the
          sharpest example: ρ ≈ 0.85, CKA ≈ 0.34. The presence of these
          cells is the single biggest reason ρ alone is insufficient.
        </li>
      </ul>

      <p className="mt-4">
        That reframing isn't deflation. Kantamneni &amp; Tegmark's
        Pythia/Latin / Llama/Latin / GPT-J/Latin results survive a wider
        protocol cleanly. The replication across Llama, Gemma, OLMo, and
        Qwen on Latin shows the helix is an architecturally general
        object. The Babylonian base-60 result is a genuinely interesting
        statement about input pipelines, even if not about transformers.
        The diagnostic isn't meant to invalidate the literature — it's
        meant to make the strong cells legible against the noisy ones.
      </p>

      <p className="mt-4">
        <strong>The general lesson:</strong> representation geometry
        needs provenance analysis. Before interpreting a manifold as a
        transformer representation, we should ask where it first
        appears in the forward pass, whether the measurement basis can
        see the relevant periods, whether the window is long enough to
        identify them, whether the same structure is already present in
        the input pipeline, and whether the pattern of distances
        survives across layers. For number helices, ρ + CKA + the two
        spectral checks are the cheapest version of those questions.
        The Babylonian random-embedding control sharpens the warning
        further: <em>some apparent neural geometry doesn't even need
        learned components</em> — additive symbolic rendering composed
        with mean-pooling is sufficient to produce a manifold that the
        Fourier basis fits. Probe work, SAE feature interpretation, and
        any "the model represents X as a manifold" claim need to
        test rendering statistics before attributing structure to
        learning.
      </p>

      <p className="mt-4">
        Two halves of the same diagnostic kit — ρ tells you which cells
        are most likely to distinguish depth-built from inherited
        geometry, and{" "}
        <a className="text-accent underline" href="https://arxiv.org/abs/2605.01148">
          Feucht, Haklay et al. (2026)
        </a>{" "}
        (<em>Arithmetic in the Wild</em>) answer it for one cell with
        a causal toolkit. Their layer-18 base-10 addition module in
        Llama 3.1 8B reads from exactly the Fourier features the trig
        basis fits, on exactly the cell ρ marks as depth-built — and
        more strikingly, the model uses base-10 features even when the
        task has a natural non-base-10 period (months, weekdays, hours).
        On the cells where ρ says depth built the geometry, theirs is
        the natural follow-up: <em>and what does the model do with it?</em>
      </p>

      <p className="mt-4">
        The open question for either side: do the cells where depth
        genuinely builds the helix outside Latin — Gemma on Devanagari,
        Persian, Roman, with Qwen 2.5 borderline on Roman and Devanagari
        — reuse the same base-10 addition circuitry Feucht et al. found
        on Llama? Roman is the sharpest test: it's not base-10
        positional, so a base-10 module is a poor fit, yet ρ = 0.48–0.70
        across Gemma and Qwen says depth is doing real work. Either
        these models reuse base-10 circuits with extra conversion
        machinery, or they build something separate. Only the
        Goodfire-style causal toolkit can decide.
      </p>
    </section>
  );
}
