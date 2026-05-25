export function Reproducing() {
  return (
    <section className="prose-body">
      <h2 className="section-heading">Reproducing</h2>
      <pre className="rounded-md bg-ink p-4 text-[12.5px] leading-snug text-paper overflow-x-auto"><code>{`# Default: Pythia-6.9B, Latin digits, paper basis [2,5,10,100], n=100
uv run main.py

# Cross-script
uv run main.py --model meta-llama/Llama-3.1-8B --script devanagari
uv run main.py --model google/gemma-4-31B --script babylonian --sweep

# Wider window
uv run main.py --script babylonian --n_max 600 --sweep

# Extended basis with T=60
uv run main.py --script babylonian --n_max 600 --periods 2,5,10,60,100 --sweep

# Full sweep: 8 models x 12 scripts x 2 windows x 2 bases.
# Idempotent -- each combo is skipped if its fig_layer_sweep.json exists.
./run.sh
uv run aggregate.py    # refreshes _index.json + _l0_share.csv`}</code></pre>

      <p className="mt-4 text-ink-mute text-sm">
        Hardware: tested on an Apple M5 Max with 128 GB unified memory (MPS).
        Falls back to CUDA, then CPU. Total wall-time for the full sweep
        is ~6–8 hours.
      </p>
    </section>
  );
}
