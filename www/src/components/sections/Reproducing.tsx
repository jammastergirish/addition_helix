export function Reproducing() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">Reproducing</h2>
      <pre className="rounded-md bg-ink p-4 text-[12.5px] leading-snug text-paper overflow-x-auto"><code>{`# Single cell: default Pythia-6.9B, Latin digits, paper basis [2,5,10,100], n=100
uv run main.py

# Cross-script
uv run main.py --model meta-llama/Llama-3.1-8B --script devanagari
uv run main.py --model google/gemma-4-31B --script babylonian --sweep

# Native bases for non-decimal scripts
uv run main.py --script babylonian   --n_max 600  --periods 2,5,10,60,100  --sweep
uv run main.py --script binary       --n_max 1024 --periods 2,4,8,16,32,64 --sweep
uv run main.py --script hexadecimal  --n_max 1024 --periods 16,32,64,256   --sweep

# Full pipeline: 8 models x 12 scripts x main + extended-basis passes,
# then aggregate, embed_control, subspace_align (CKA), and compare.
# Idempotent -- each combo is skipped if its fig_layer_sweep.json exists.
./run.sh

# Run.sh chains:
#   pass 1     -- main sweep, paper-default basis (per (model, script))
#   extras     -- Babylonian / binary / hex native-basis runs
#   pass 4     -- aggregate.py      (writes _index.json, _l0_share.csv)
#   pass 5     -- embed_control.py  (random-embedding control)
#   pass 6     -- subspace_align.py (CKA, skip-if-exists)
#   pass 7     -- compare.py        (cross-model comparison PNG)

# Filter to one model family (matches HF id substring):
./run.sh gpt-j
./run.sh gemma`}</code></pre>

      <p className="mt-4 text-ink-mute text-sm">
        Hardware: tested on an Apple M5 Max with 128 GB unified memory
        (MPS). Falls back to CUDA, then CPU. Total wall-time for the
        full pipeline (8 models, all passes) is ~8–12 hours; most of
        that is the main sweep and <code>subspace_align.py</code> (which
        reloads each model for forward passes at layer 0 and the peak
        layer).
      </p>
    </section>
  );
}
