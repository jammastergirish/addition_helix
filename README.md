# addition_helix

A from-scratch replication of **Kantamneni & Tegmark, _"Language Models Use
Trigonometry to Do Addition"_** ([arXiv:2502.00873](https://arxiv.org/abs/2502.00873),
2025), with an extension that lets you swap the numeral script the model
sees (Latin, Arabic-Indic, Persian, Devanagari, Chinese-positional, Greek
alphabetic, or Roman).

## The claim, in one paragraph

For an integer `a ∈ [0, 99]`, the residual stream of a transformer LM
encodes `a` as a **generalised helix** — one linear "number-line" axis
plus four orthogonal modular circles of period `T ∈ {2, 5, 10, 100}`.
T=2 captures parity, T=5 captures `a mod 5`, T=10 captures the units
digit, T=100 captures coarse position. Addition is performed by
*rotating* these circles — the paper's "Clock" algorithm.

## What `main.py` does

1. Pulls the residual stream `h(a)` for every `a ∈ [0, 99]` at a middle
   layer.
2. **FFTs** each hidden dimension across `a` — peaks at `f = 1/2, 1/5,
   1/10, 1/100` should emerge from the data (no hand-tuning).
3. **Fits** the helix model `h(a) ≈ Wᵀ B(a)` by linear regression and
   reports R².
4. Saves three figures matching paper Figures 1–3:

| file | paper figure | content |
|---|---|---|
| `fig2_fourier_pc1.png` | Fig 2 | FFT spectrum (top) + PC1-vs-`a` linear ramp (bottom) |
| `fig3_circles_and_line.png` | Fig 3 | four `(cos_T, sin_T)` panels + linear "number line" strip |
| `fig1_helix_T10.png` | Fig 1 (right) | the iconic 3D `T=10` helix |

## Requirements

- Python ≥ 3.10
- `uv` (handles the PEP 723 inline script metadata at the top of
  `main.py` automatically — no separate venv needed)
- A Hugging Face token in `.env` as `HF_TOKEN=…` (needed for gated
  models like Llama 3)
- Hardware: defaults to Apple MPS, falls back to CUDA, then CPU. With
  128 GB unified memory either Pythia-6.9B or Llama-3.1-8B fits in bf16.

## Run

```bash
# default: Pythia-6.9B, Latin digits, mean-pooled
uv run main.py

# pick a model
uv run main.py --model meta-llama/Llama-3.1-8B
uv run main.py --model google/gemma-4-E4B
uv run main.py --model google/gemma-4-31B

# pick a numeral script
uv run main.py --script arabic
uv run main.py --script chinese
uv run main.py --script roman

# pick a pooling mode (see "pooling" below)
uv run main.py --pool last       # paper's default for single-token Latin
uv run main.py --pool mean       # the default; honest for multi-token numerals
```

Or run the full sweep — `run.sh` covers four models × seven scripts:

```bash
./run.sh
```

The models in the sweep:

| model | params | notes |
|---|---|---|
| `EleutherAI/pythia-6.9b` | 6.9 B | paper's primary model; single-token Latin digits |
| `meta-llama/Llama-3.1-8B` | 8 B | digit-by-digit tokenization for all scripts |
| `google/gemma-4-E4B` | ~4 B (efficient) | smaller, fast turn-around |
| `google/gemma-4-31B` | 31 B | largest; ~62 GB in bf16 — comfortable on 128 GB unified memory |

All are **base** (non-instruction-tuned) checkpoints, matching the paper's setup.

## Numeral scripts

| script | example (23) | family | helix predicted? |
|---|---|---|---|
| `latin` (default) | `23` | positional base-10 | yes (paper's setup) |
| `arabic` | `٢٣` | positional base-10, Arabic-Indic | yes |
| `persian` | `۲۳` | positional base-10, Persian | yes |
| `devanagari` | `२३` | positional base-10, Hindi | yes |
| `chinese` | `二三` | positional base-10, CJK glyphs | yes |
| `greek` | `κγ` | additive (Milesian) | predicted to vanish |
| `roman` | `XXIII` | additive | predicted to vanish |

## Pooling (`--pool`)

For Pythia, Latin numbers 0–99 are single BPE tokens. Reading the
residual stream "at the operand's last position" is unambiguous. For
every other script (or for Latin on Llama 3, which tokenizes
digit-by-digit), a numeral spans several sub-tokens and the choice of
pool matters:

- **`--pool last`** reads the activation at the very last sub-token.
  For `٢٣` that's literally the second byte of `٣` — and `23, 33, 43, …`
  all end in `٣`, so they all read the same thing. This bakes a
  period-10 cycle into the data **by construction** and produces fake
  "helix-like" FFT peaks for reasons unrelated to anything the model
  learned.

- **`--pool mean`** (default) averages activations across all
  sub-tokens of the numeral. Every position contributes equally; the
  last-byte artifact disappears. This is the honest test for non-Latin
  scripts.

For single-token regimes (Latin on Pythia) `mean` and `last` produce
identical activations.

## Output layout

```
out/<model_with_slashes_replaced>/<script>/<pool>/
    ├── fig1_helix_T10.png
    ├── fig2_fourier_pc1.png
    └── fig3_circles_and_line.png
```

For example: `out/EleutherAI__pythia-6.9b/latin/mean/fig1_helix_T10.png`.

This nesting means re-running with different `--script` / `--pool` /
model values leaves prior outputs intact and lets you compare side by
side.

## Console output you should read

For each run the script prints:

```
helix fit R^2 (variance-weighted) = 0.46
  9-d PCA reconstruction R^2      = 0.55
```

The 9-D PCA R² is the **upper bound** on what any 9-parameter fit can
capture. If `helix R² / PCA R²` is close to 1 (above ~0.8) the helix
basis is essentially the 9-D structure of the residual stream — the
paper's main quantitative claim. For non-positional scripts this ratio
typically drops well below 0.7.

## Citation

```
@article{kantamneni2025helix,
  title  = {Language Models Use Trigonometry to Do Addition},
  author = {Kantamneni, Subhash and Tegmark, Max},
  journal= {arXiv preprint arXiv:2502.00873},
  year   = {2025}
}
```
