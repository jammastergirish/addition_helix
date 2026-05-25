# From Latin Digits to Babylonian Cuneiform

A from-scratch extension of Kantamneni & Tegmark, *"Language Models Use
Trigonometry to Do Addition"* ([arXiv:2502.00873](https://arxiv.org/abs/2502.00873)).
The paper shows transformer LMs encode integers as a "generalised helix"
(linear axis + modular circles); this project asks **where the helix
comes from** — depth, the input pipeline (rendering / tokenization /
embedding / pooling), or the measurement protocol itself.

The diagnostic kit sweeps **8 models × 12 numeral scripts** (Latin digits
through Babylonian cuneiform) and reports ρ = helix R²(L=0) / helix R²(peak),
a CKA representation-alignment check, and a random-embedding control.

Write-up: `www/` is the interactive React site.
Paper draft: `paper.pdf`.

## Layout

```
helix_lib.py        shared renderers / helix basis / fit / config helpers
main.py             per-(model, script) sweep — collects activations,
                    fits the helix, dumps figures + per-layer JSON
aggregate.py        walks out/, builds out/_index.json for the React site
embed_control.py    random-embedding control across models / scripts
subspace_align.py   linear CKA between L=0 and peak-layer helix subspaces
compare.py          stacks the per-model layer-sweep PNGs into one image
run.sh              driver: sweeps every (model, script, basis) combo
www/                React + Vite blogpost site (reads out/_index.json)
out/                generated data and figures (gitignored)
```

## Requirements

- Python ≥ 3.10 with [`uv`](https://docs.astral.sh/uv/) (each script
  declares its own deps via PEP 723 inline metadata — no separate venv).
- Node ≥ 20 for the React site.
- A Hugging Face token in `.env` as `HF_TOKEN=…` (Llama 3, Gemma 4 are
  gated).
- bash ≥ 4 for `run.sh` (`brew install bash` on macOS).
- GPU optional — the code prefers Apple MPS, then CUDA, then CPU.

## Run

```bash
# one cell
uv run main.py --model meta-llama/Llama-3.1-8B --script latin --sweep

# full sweep: 8 models × 12 scripts + extended-basis runs (~6–8 h)
./run.sh

# aggregate + auxiliary diagnostics (after the sweep)
uv run aggregate.py
uv run embed_control.py
uv run subspace_align.py --scripts all

# the React blogpost
cd www && npm install && npm run dev
```

`./run.sh` is idempotent — any combo whose `fig_layer_sweep.json` already
exists is skipped. Filter with a substring: `./run.sh gemma`, `./run.sh 31B`.

## Citation

```
@article{kantamneni2025helix,
  title  = {Language Models Use Trigonometry to Do Addition},
  author = {Kantamneni, Subhash and Tegmark, Max},
  journal= {arXiv preprint arXiv:2502.00873},
  year   = {2025}
}
```
