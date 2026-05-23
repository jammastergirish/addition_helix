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
        the paper's "Clock" half of the argument, which we don't replicate
        here. We leave that to a follow-up.
      </p>
    </section>
  );
}
