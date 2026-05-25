export function Conclusion() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">What this changes</h2>

      <p>
        The contribution is the diagnostic kit, not the matrix. The
        standard helix-R² protocol has three failure modes that can make
        it answer the wrong question — two hide real structure (basis
        bandwidth, window wrap), one misattributes pre-transformer
        structure to depth (ρ). Add the CKA extension and a fourth
        nuance appears: high ρ doesn't always mean inheritance — it can
        also mean "depth rebuilds a similar-score helix at a different
        subspace." Putting all four together (basis, window, ρ, CKA)
        across 56 cells gives this picture:
      </p>

      <ul className="mt-3 space-y-2 list-disc pl-6">
        <li>
          <strong>Pass-through cells</strong> (high ρ AND high CKA — the
          tokenizer/embedding front end supplies the helix and depth
          doesn't touch it): Qwen-32B/Latin, OLMo/Latin, most
          Pythia/Llama non-Latin positional cells, Qwen on Chinese/Greek.
          The clearest existence proof of pre-transformer inheritance.
        </li>
        <li>
          <strong>Depth-amplification cells</strong> (low-to-moderate ρ
          AND high CKA — the L=0 helix is real and depth refines it in
          the same directions): Pythia/Latin, Llama/Latin, GPT-J/Latin,
          Roman on almost every model, and{" "}
          <strong>hexadecimal on the older research models</strong>{" "}
          (Pythia, Llama, GPT-J, OLMo) — almost certainly driven by
          code training. Hex is the second-cleanest depth-built case in
          the matrix.
        </li>
        <li>
          <strong>Mechanical cells</strong> (learned-vs-random gap ≈ 0
          at L=0): Babylonian on every model, <em>and</em> binary on
          every model. Both cases share the same underlying mechanism —
          additive symbolic rendering (wedge counts, bit counts)
          composed with mean-pooling produces vectors that vary smoothly
          with <em>n</em> regardless of what's in the embedding table.
          The structure is in the renderer + pooling alone.
        </li>
        <li>
          <strong>Rebuild cells</strong> (high ρ but LOW CKA — depth
          produces a similar-score helix at a different subspace,
          flattening to "inherited" under ρ alone): the whole Gemma
          family on most scripts, Babylonian on Pythia/Llama/OLMo,
          Qwen on Devanagari/Thai. The presence of these cells is the
          single biggest reason ρ alone is insufficient.
        </li>
      </ul>

      <p className="mt-4">
        Without these diagnostics, the cross-script and Babylonian
        measurements would read as confirmation of the original paper's
        central claim. With them, "the helix replicates on script S in
        model M" splits into at least four mechanistically distinct
        statements, only some of which support the "transformer
        represents number geometrically" reading. Qwen 2.5/Latin is the
        sharpest existence proof of pre-transformer inheritance: helix
        R² ≈ 0.54, ρ ≈ 1, CKA ≈ 1. Gemma/Latin is the sharpest
        existence proof of the rebuild regime: ρ ≈ 0.85, CKA ≈ 0.34.
      </p>

      <p className="mt-4">
        That reframing isn't deflation. The paper's headline finding on
        Pythia/Latin survives a wider protocol cleanly. The replication
        across Llama, Gemma, OLMo, and Qwen on Latin shows the helix is
        an architecturally general object. The Babylonian base-60 result
        is a genuinely interesting statement about tokenisers, even if
        not about transformers. The point of the diagnostic isn't to
        invalidate the literature — it's to make the strong cells
        legible against the noisy ones.
      </p>

      <h3 className="section-subheading">Feucht &amp; Haklay as natural complement</h3>

      <p>
        Two halves of the same diagnostic kit. ρ tells you{" "}
        <em>which cells are most likely to distinguish depth-built
        geometry from inherited geometry before doing causal work</em>;
        Feucht, Haklay et al. (
        <a className="text-accent underline" href="https://arxiv.org/abs/2605.01148">
          arXiv:2605.01148
        </a>,{" "}
        <em>Arithmetic in the Wild: Llama uses Base-10 Addition to Reason
        About Cyclic Concepts</em>, May 2026) answer it for Llama 3.1 8B.
        Their layer-18 base-10 addition module reads from exactly the
        Fourier features the trig basis fits, on exactly the cell my ρ
        marks as depth-built.
      </p>

      <p className="mt-3">
        Their headline finding is sharper than "the circles are used":
        Llama uses <strong>base-10 Fourier features</strong> (periods 2,
        5, 10, 20, 50, 100) even for tasks with natural non-base-10
        periods. For "six months after August" the model converts August
        to 8, computes 6+8=14 using base-10 addition at layer 18, then
        maps 14 back to February in later layers. The 12-month circular
        geometry exists at the <em>input</em> but is not what layer 18
        reads. They isolate 28 MLP neurons doing the addition,
        partitioned by Fourier period, and confirm causality with
        DAS-based steering.
      </p>

      <p className="mt-3">
        Two consequences for this post:
      </p>

      <ul className="mt-2 space-y-2 list-disc pl-6">
        <li>
          The K&amp;T-style basis I fit ({"{"}2, 5, 10, 100{"}"}) isn't
          just a convenient measurement target — on Llama at least, the
          same kind of base-10 Fourier features really are the substrate
          arithmetic and arithmetic-like reasoning runs on. Strong
          vindication that on the cells where ρ says depth built the
          geometry, it's a real computational object.
        </li>
        <li>
          ρ measures <em>where the geometry is built</em>, not whether
          it's read. On cells with ρ ≈ 1 (Pythia/Llama × non-Latin
          positional, all Babylonian), the inherited circles could
          still in principle be read by a downstream addition module —
          many of those scripts (Arabic-Indic, Devanagari, CJK
          positional) <em>are</em> base-10 and may tokenize per-digit. ρ
          tells us the geometry-building work isn't happening in the
          transformer; their toolkit is what would answer whether the
          geometry is read regardless of where it was built.
        </li>
      </ul>

      <h3 className="section-subheading">The point</h3>
      <p>
        The lesson is not that helices are artifacts. The strongest
        cells remain strong, and recent causal work shows that Fourier
        number features can be real computational substrates. The
        lesson is that representation geometry needs provenance
        analysis. Before interpreting a manifold as a transformer
        representation, we should ask where it first appears in the
        forward pass, whether the measurement basis can see the
        relevant periods, whether the window is long enough to identify
        them, and whether the same structure is already present in the
        tokenizer/embedding front end. For number helices, ρ plus the
        two spectral checks is the cheapest version of that provenance
        question. It is also the question every "the model represents X
        as Y manifold" claim should be asked.
      </p>

      <p className="mt-4">
        The Babylonian random-embedding control pushes that lesson one
        layer deeper. <strong>Some apparent neural geometry doesn't even
        need learned components</strong>: smooth, low-dimensional,
        Fourier-fittable manifolds can emerge from the input pipeline
        itself — additive symbolic rendering composed with mean-pooling
        is sufficient to produce structure that the Fourier basis fits
        and that an interpretability paper could publish as "the model
        represents cuneiform numbers geometrically." Probe work, SAE
        feature interpretation, and any "the model represents X as a
        manifold" claim need to test rendering statistics before
        attributing structure to learning — the same way differential
        diagnosis needs to rule out the obvious before reaching for the
        interesting.
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
