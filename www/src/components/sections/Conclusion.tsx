export function Conclusion() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">What this changes</h2>

      <p>
        The original paper's headline finding on Pythia/Latin survives a
        wider protocol cleanly: Llama matches it, Gemma and OLMo replicate
        the geometry on Latin, and the helix appears in scripts the model
        was trained on. But the <em>generality</em> — "the helix is how
        transformers encode quantity" — is overstated by the cross-script
        and Babylonian measurements once you separate what the tokeniser
        supplies from what depth builds. On most non-Latin cells, the trig
        basis is fitting structure the tokeniser already put there.
      </p>

      <p>
        The natural next question — <em>whether the helix subspace is the
        representation downstream computation actually reads</em>, on the
        cells where depth genuinely builds it — requires causal intervention
        at the <code>=</code> position, not at the operand position. That's
        the paper's "Clock" half of the argument, which I don't replicate
        here.
      </p>

      <p>
        Feucht, Haklay et al. (
        <a className="text-accent underline" href="https://arxiv.org/abs/2605.01148">
          arXiv:2605.01148
        </a>,{" "}
        <em>Arithmetic in the Wild: Llama uses Base-10 Addition to Reason
        About Cyclic Concepts</em>, May 2026) published exactly that
        follow-up on Llama 3.1 8B — one of the five models tested here.
        Their headline finding is sharper than "the circles are used":
        Llama uses <strong>base-10 Fourier features</strong> (periods 2,
        5, 10, 20, 50, 100) to compute even for tasks with natural
        non-base-10 periods. For "six months after August" the model
        doesn't rotate around a 12-circle for months — it converts August
        to 8, computes 6+8=14 using base-10 addition at layer 18, then
        maps 14 back to February in later layers. They isolate 28 MLP
        neurons performing the addition, partition them by Fourier
        period, and confirm causality with DAS-based steering. The two
        results fit together cleanly:
      </p>

      <ul className="mt-2 space-y-2 list-disc pl-6">
        <li>
          <strong>Where the two works agree.</strong> The K&amp;T-style
          basis I fit ({"{"}2, 5, 10, 100{"}"}) isn't just a convenient
          measurement target — on Llama at least, the same kind of
          base-10 Fourier features really are the substrate Llama uses
          for arithmetic, and for arithmetic-like reasoning over months,
          weekdays, and hours. Strong vindication that this geometry is a
          real computational object on at least one of my five models.
        </li>
        <li>
          <strong>Where their result tightens the framing for mine.</strong>{" "}
          Without their causal evidence, my ρ diagnostic could be
          dismissed as "but maybe even the depth-built helix isn't used
          downstream." On Llama/Latin (ρ = 0.39) that escape is now
          closed — the structure measurably built by depth is also
          measurably read by depth.
        </li>
        <li>
          <strong>Where my measurement refines what their result implies.</strong>{" "}
          ρ tells you <em>where</em> the helix is built, not whether it's
          used. So on cells with ρ ≈ 1 (Pythia/Llama × non-Latin
          positional, all Babylonian, Greek), I can't rule out that the
          tokeniser-inherited circles are still read by a downstream
          base-10 addition module — many of those scripts (Arabic-Indic,
          Devanagari, Chinese-positional) <em>are</em> base-10 and may
          tokenise per-digit. What ρ does say is that the
          <em>geometry-building</em> work isn't happening in the
          transformer there; whether the geometry is used downstream is a
          separate question, and Goodfire's toolkit is what answers it.
        </li>
      </ul>

      <p className="mt-4">
        The open question for either side: do the cells where depth
        genuinely builds the helix outside Latin (Gemma on Devanagari,
        Persian, Roman) reuse the same base-10 addition-module circuitry
        Feucht et al. identified on Llama — or do they build
        script-specific machinery? Roman is especially interesting: it's
        not base-10 positional, so a base-10 addition module is a poor
        fit, yet Gemma's ρ = 0.48–0.55 there says depth is doing real
        work. Either Gemma reuses base-10 circuits with extra
        conversion machinery, or it builds a separate additive module —
        and only Goodfire-style causal tooling can tell which.
      </p>
    </section>
  );
}
